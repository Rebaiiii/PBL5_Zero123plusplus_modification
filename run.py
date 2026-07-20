import os
import argparse
import sys
import warnings
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import generate_zero123plus_candidate, remove_background, resize_foreground, save_video
from zero123plus.reference_adapter import ReferenceAdapter, extract_reference_adapter_state_dict
from zero123plus.reference_utils import (
    load_reference_images as load_adapter_reference_images,
    references_to_pil,
    references_to_slot_weights,
    references_to_view_ids,
)


def load_reference_adapter_checkpoint(adapter, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    adapter_state = extract_reference_adapter_state_dict(checkpoint)
    incompatible = adapter.load_state_dict(adapter_state, strict=False)
    adapter.to(device=device)
    return incompatible


def reference_adapter_weight_norms(adapter):
    ref_proj_sq = 0.0
    view_embed_sq = 0.0
    for name, param in adapter.named_parameters():
        value = float(param.detach().float().pow(2).sum().item())
        if name.startswith("ref_proj"):
            ref_proj_sq += value
        elif name.startswith("view_embed"):
            view_embed_sq += value
    return ref_proj_sq ** 0.5, view_embed_sq ** 0.5


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames


###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
parser.add_argument('--reference_images', '--rag_refs', dest='reference_images', type=str, default=None, help='Folder or image file containing references.')
parser.add_argument('--reference_view_labels', '--rag_view_labels', dest='reference_view_labels', type=str, default=None, help='Optional JSON mapping reference filenames to front, side, back, or unknown.')
parser.add_argument('--reference_metadata', '--rag_ref_metadata', dest='reference_metadata', type=str, default=None, help='JSON mapping reference filenames to azimuth and elevation.')
parser.add_argument('--reference_max_size', '--rag_max_size', dest='reference_max_size', type=int, default=1024, help='Maximum side length used when loading local reference images.')
parser.add_argument('--enable_reference_adapter', '--enable_rag_adapter', dest='enable_reference_adapter', action='store_true', help='Enable the experimental view-aware reference token adapter.')
parser.add_argument('--reference_adapter_ckpt', '--rag_adapter_ckpt', dest='reference_adapter_ckpt', type=str, default=None, help='Path to a trained reference adapter checkpoint or state_dict.')
parser.add_argument('--reference_token_scale', '--rag_token_scale', dest='reference_token_scale', type=float, default=0.1, help='Scale for routed reference adapter slot tokens.')
parser.add_argument('--reference_global_scale', '--rag_global_scale', dest='reference_global_scale', type=float, default=0.05, help='Scale for weak global reference identity token.')
parser.add_argument('--reference_global_token_enabled', '--rag_global_reference_token_enabled', dest='reference_global_token_enabled', action=argparse.BooleanOptionalAction, default=True, help='Enable the weak global valid-reference token.')
parser.add_argument('--reference_match_scale', '--rag_match_scale', dest='reference_match_scale', type=float, default=1.0, help='Reference adapter exact-slot routing weight.')
parser.add_argument('--reference_near_scale', '--rag_near_scale', dest='reference_near_scale', type=float, default=0.35, help='Reference adapter nearby-slot routing weight.')
parser.add_argument('--reference_nonmatch_scale', '--rag_nonmatch_scale', dest='reference_nonmatch_scale', type=float, default=0.05, help='Reference adapter nonmatching-slot routing weight.')
parser.add_argument('--reference_unknown_scale', '--rag_unknown_scale', dest='reference_unknown_scale', type=float, default=0.1, help='Reference adapter unknown-view routing weight.')
parser.add_argument('--reference_spatial_gating', '--rag_spatial_gating', dest='reference_spatial_gating', action='store_true', help='Spatially gate reference adapter influence using fixed Zero123++ latent tile masks.')
parser.add_argument('--reference_spatial_gate_scale', '--rag_spatial_gate_scale', dest='reference_spatial_gate_scale', type=float, default=1.0, help='Scale for the fixed spatial reference tile gate.')
parser.add_argument('--zero123plus_pose_version', choices=['v1.1', 'v1.2'], default='v1.2', help='Zero123++ fixed target pose version used for reference pose-slot assignment.')
parser.add_argument('--reference_azimuth_sigma', '--rag_ref_azimuth_sigma', dest='reference_azimuth_sigma', type=float, default=45.0, help='Azimuth sigma for pose-aware reference slot weights.')
parser.add_argument('--reference_elevation_sigma', '--rag_ref_elevation_sigma', dest='reference_elevation_sigma', type=float, default=25.0, help='Elevation sigma for pose-aware reference slot weights.')
parser.add_argument('--reference_slot_weight_mode', '--rag_slot_weight_mode', dest='reference_slot_weight_mode', choices=['local', 'wide'], default='local', help='Reference-to-output-slot routing mode.')
parser.add_argument('--reference_slot_weight_sigma_deg', '--rag_slot_weight_sigma_deg', dest='reference_slot_weight_sigma_deg', type=float, default=80.0, help='Wide routing circular Gaussian sigma in degrees.')
parser.add_argument('--reference_slot_weight_min', '--rag_slot_weight_min', dest='reference_slot_weight_min', type=float, default=0.05, help='Minimum reference slot weight.')
parser.add_argument('--reference_slot_weight_normalize', '--rag_slot_weight_normalize', dest='reference_slot_weight_normalize', action=argparse.BooleanOptionalAction, default=True, help='Normalize wide routing weights to peak at 1.0.')
parser.add_argument('--reference_slot_weight_elevation_weight', '--rag_slot_weight_elevation_weight', dest='reference_slot_weight_elevation_weight', type=float, default=0.25, help='Elevation contribution for wide routing distance.')
parser.add_argument('--reference_cross_view_propagation_enabled', '--rag_cross_view_propagation_enabled', dest='reference_cross_view_propagation_enabled', action='store_true', help='Spread part of reference routing to nearby output slots.')
parser.add_argument('--reference_cross_view_propagation_strength', '--rag_cross_view_propagation_strength', dest='reference_cross_view_propagation_strength', type=float, default=0.3, help='Strength for cross-view slot propagation.')
parser.add_argument('--reference_cross_view_neighbor_degrees', '--rag_cross_view_neighbor_degrees', dest='reference_cross_view_neighbor_degrees', type=float, default=120.0, help='Azimuth range for cross-view slot propagation.')
parser.add_argument('--reference_train_adapter_only', '--rag_train_adapter_only', dest='reference_train_adapter_only', action='store_true', help='Training config flag placeholder: train only reference adapter modules.')
parser.add_argument('--reference_unfreeze_crossattn', '--rag_unfreeze_crossattn', dest='reference_unfreeze_crossattn', action='store_true', help='Training config flag placeholder: selectively unfreeze cross-attention.')
parser.add_argument('--reference_debug_dump', '--rag_debug_dump', dest='reference_debug_dump', action='store_true', help='Print reference adapter debug info.')
parser.add_argument('--mesh_grid_res', type=int, default=None, help='Override FlexiCubes mesh grid resolution to reduce mesh extraction memory.')
parser.add_argument('--render_resolution', type=int, default=None, help='Override render resolution to reduce rendering memory.')
parser.add_argument('--texture_resolution', type=int, default=None, help='Override texture resolution to reduce texture export memory.')

for raw_argument in sys.argv[1:]:
    option = raw_argument.split('=', 1)[0]
    if 'rag' not in option:
        continue
    for action in parser._actions:
        if option in action.option_strings:
            warnings.warn(
                f"{option} is deprecated; use {action.option_strings[0]} instead.",
                FutureWarning,
                stacklevel=1,
            )
            break

args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config
if args.mesh_grid_res is not None and hasattr(model_config, "params") and "grid_res" in model_config.params:
    model_config.params.grid_res = args.mesh_grid_res
if args.render_resolution is not None:
    infer_config.render_resolution = args.render_resolution
if args.texture_resolution is not None:
    infer_config.texture_resolution = args.texture_resolution

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
print('Loading custom white-background unet ...')
if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)
reference_adapter = None
if args.enable_reference_adapter:
    embed_dim = getattr(pipeline.vision_encoder.config, "projection_dim", None)
    if embed_dim is None:
        embed_dim = getattr(pipeline.vision_encoder.config, "hidden_size", None)
    if embed_dim is None:
        raise ValueError("Could not infer CLIP vision embedding dimension for reference adapter.")
    reference_adapter = ReferenceAdapter(embed_dim=embed_dim).to(device=device, dtype=torch.float16)
    reference_adapter_loaded = False
    missing_keys = []
    unexpected_keys = []
    if args.reference_adapter_ckpt:
        if not os.path.exists(args.reference_adapter_ckpt):
            raise FileNotFoundError(f"reference adapter checkpoint not found: {args.reference_adapter_ckpt}")
        incompatible = load_reference_adapter_checkpoint(reference_adapter, args.reference_adapter_ckpt, device)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        reference_adapter_loaded = True
    reference_adapter.eval()
    print(f"[REFERENCE-ADAPTER] enabled with embed_dim={embed_dim}")
    if not reference_adapter_loaded:
        print("[REFERENCE-ADAPTER] WARNING: no --reference_adapter_ckpt provided; using random/default adapter weights.")
    if args.reference_debug_dump:
        ref_proj_norm, view_embed_norm = reference_adapter_weight_norms(reference_adapter)
        print(f"[REFERENCE-ADAPTER] adapter checkpoint loaded: {reference_adapter_loaded}")
        print(f"[REFERENCE-ADAPTER] checkpoint path: {args.reference_adapter_ckpt}")
        print(f"[REFERENCE-ADAPTER] missing keys: {missing_keys}")
        print(f"[REFERENCE-ADAPTER] unexpected keys: {unexpected_keys}")
        print(f"[REFERENCE-ADAPTER] ref_proj weight norm: {ref_proj_norm:.6f}")
        print(f"[REFERENCE-ADAPTER] view_embed weight norm: {view_embed_norm:.6f}")
        print(
            "[REFERENCE-ADAPTER] weights status: "
            f"{'loaded_from_checkpoint' if reference_adapter_loaded else 'random_default_untrained'}"
        )

# make output directories
image_path = os.path.join(args.output_path, config_name, 'images')
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
video_path = os.path.join(args.output_path, config_name, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# process input files
if os.path.isdir(args.input_path):
    input_files = [
        os.path.join(args.input_path, file) 
        for file in os.listdir(args.input_path) 
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
    ]
else:
    input_files = [args.input_path]
print(f'Total number of input images: {len(input_files)}')


###############################################################################
# Stage 1: Multiview generation.
###############################################################################

rembg_session = None if args.no_rembg else rembg.new_session()

outputs = []
for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).split('.')[0]
    print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')

    # remove background optionally
    input_image = Image.open(image_file)
    if not args.no_rembg:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    # sampling
    use_reference_adapter = args.enable_reference_adapter and args.reference_images is not None
    if use_reference_adapter:
        references = load_adapter_reference_images(
            args.reference_images,
            view_labels_path=args.reference_view_labels,
            metadata_path=args.reference_metadata,
            pose_version=args.zero123plus_pose_version,
            max_image_size=args.reference_max_size,
        )
        if not references:
            raise ValueError(f"No reference images found for reference adapter: {args.reference_images}")
        print(f"[REFERENCE-ADAPTER] input={image_file}")
        for reference in references:
            print(
                "[REFERENCE-ADAPTER] "
                f"reference={reference.path} "
                f"view_label={reference.view_label} "
                f"view_id={reference.view_id} "
                f"azimuth={reference.azimuth} "
                f"elevation={reference.elevation} "
                f"pose_source={reference.pose_source}"
            )
        known_reference_count = sum(
            reference.azimuth is not None and reference.elevation is not None
            for reference in references
        )
        if known_reference_count == 0:
            print(
                "[REFERENCE-ADAPTER] WARNING: no azimuth/elevation metadata or parseable filenames found; "
                "routing will be weak/unknown."
            )
        elif known_reference_count < len(references):
            print(
                f"[REFERENCE-ADAPTER] WARNING: only {known_reference_count}/{len(references)} references "
                "have azimuth/elevation; remaining references use weak/coarse routing."
            )
        reference_slot_weights = references_to_slot_weights(
            references,
            pose_version=args.zero123plus_pose_version,
            azimuth_sigma=args.reference_azimuth_sigma,
            elevation_sigma=args.reference_elevation_sigma,
            mode=args.reference_slot_weight_mode,
            sigma_deg=args.reference_slot_weight_sigma_deg,
            min_weight=args.reference_slot_weight_min,
            normalize=args.reference_slot_weight_normalize,
            elevation_weight=args.reference_slot_weight_elevation_weight,
            cross_view_propagation_enabled=args.reference_cross_view_propagation_enabled,
            cross_view_propagation_strength=args.reference_cross_view_propagation_strength,
            cross_view_neighbor_degrees=args.reference_cross_view_neighbor_degrees,
        )
        print("[REFERENCE-ADAPTER] method = manual_metadata_spatial_gating")
        print("[REFERENCE-ADAPTER] auto_view_assignment = disabled")
        print(f"[REFERENCE-ADAPTER] checkpoint_loaded = {str(reference_adapter_loaded).lower()}")
        print(f"[REFERENCE-ADAPTER] references_known = {known_reference_count}/{len(references)}")
        print(f"[REFERENCE-ADAPTER] slot_weight_mode = {args.reference_slot_weight_mode}")
        print(f"[REFERENCE-ADAPTER] cross_view_propagation = {str(args.reference_cross_view_propagation_enabled).lower()}")
        print(f"[REFERENCE-ADAPTER] spatial_gating = {str(args.reference_spatial_gating).lower()}")
        print(f"[REFERENCE-ADAPTER] ref_slot_weights = {reference_slot_weights}")
        output_image = generate_zero123plus_candidate(
            pipeline,
            input_image,
            args.diffusion_steps,
            device,
            args.seed,
            reference_images=references_to_pil(references),
            reference_view_ids=references_to_view_ids(references),
            reference_slot_weights=reference_slot_weights,
            reference_adapter=reference_adapter,
            reference_token_scale=args.reference_token_scale,
            reference_global_scale=args.reference_global_scale,
            reference_global_token_enabled=args.reference_global_token_enabled,
            reference_match_scale=args.reference_match_scale,
            reference_near_scale=args.reference_near_scale,
            reference_nonmatch_scale=args.reference_nonmatch_scale,
            reference_unknown_scale=args.reference_unknown_scale,
            reference_spatial_gating=args.reference_spatial_gating,
            reference_spatial_gate_scale=args.reference_spatial_gate_scale,
            reference_debug_dump=args.reference_debug_dump,
        )
    else:
        output_image = generate_zero123plus_candidate(
            pipeline,
            input_image,
            args.diffusion_steps,
            device,
            args.seed,
        )

    output_image.save(os.path.join(image_path, f'{name}.png'))
    print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    outputs.append({'name': name, 'images': images})

# delete pipeline to save memory
del pipeline
torch.cuda.empty_cache()

###############################################################################
# Stage 2: Reconstruction.
###############################################################################

# Load the reconstruction model only after Zero123++ generation is complete so
# both large models do not occupy GPU memory at the same time.
print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_filename = os.path.basename(infer_config.model_path)
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=model_ckpt_filename, repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras = input_cameras[:, indices]

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get mesh
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )
        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")

        # get video
        if args.save_video:
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.render_resolution
            render_cameras = get_render_cameras(
                batch_size=1, 
                M=120, 
                radius=args.distance, 
                elevation=20.0,
                is_flexicubes=IS_FLEXICUBES,
            ).to(device)
            
            frames = render_frames(
                model, 
                planes, 
                render_cameras=render_cameras, 
                render_size=render_size, 
                chunk_size=chunk_size, 
                is_flexicubes=IS_FLEXICUBES,
            )

            save_video(
                frames,
                video_path_idx,
                fps=30,
            )
            print(f"Video saved to {video_path_idx}")
