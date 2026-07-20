from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import v2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.camera_util import get_zero123plus_input_cameras
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.train_util import instantiate_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct user-study result sheets into OBJ and Blender files."
    )
    parser.add_argument("--config", default="configs/instant-mesh-large-lowvram.yaml")
    parser.add_argument("--study_dir", default="outputs/user_study")
    parser.add_argument("--blender_path", default=r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--view", type=int, choices=(4, 6), default=6)
    parser.add_argument("--mesh_grid_res", type=int, default=None)
    parser.add_argument("--render_resolution", type=int, default=None)
    parser.add_argument("--texture_resolution", type=int, default=None)
    parser.add_argument("--export_texmap", action="store_true")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only_object", default=None)
    parser.add_argument("--only_result", choices=("result_1", "result_2", "result_3"), default=None)
    return parser.parse_args()


def load_manifest(study_dir: Path) -> dict:
    manifest_path = study_dir / "study_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Study manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8-sig"))


def iter_result_jobs(study_dir: Path, manifest: dict, only_object: str | None, only_result: str | None):
    public_root = study_dir / "public_stimuli"
    if not public_root.exists():
        public_root = study_dir / "objects"
    if not public_root.exists():
        raise FileNotFoundError(f"Study object folder not found: {study_dir / 'public_stimuli'} or {study_dir / 'objects'}")
    for object_entry in manifest["objects"]:
        object_id = object_entry["object_id"]
        if only_object and object_id != only_object:
            continue
        for result_entry in object_entry["results"]:
            blind_result = result_entry["blind_result"]
            if only_result and blind_result != only_result:
                continue
            sheet_path = public_root / object_id / result_entry["image"]
            model_dir = public_root / object_id / "results" / blind_result / "model"
            yield {
                "object_id": object_id,
                "blind_result": blind_result,
                "sheet_path": sheet_path,
                "model_dir": model_dir,
                "name": f"{object_id}_{blind_result}",
            }


def load_sheet(sheet_path: Path) -> torch.Tensor:
    if not sheet_path.exists():
        raise FileNotFoundError(f"Zero123++ sheet not found: {sheet_path}")
    image = Image.open(sheet_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float()
    return rearrange(tensor, "c (n h) (m w) -> (n m) c h w", n=3, m=2)


def obj_to_blend(blender_path: str, obj_path: Path, blend_path: Path) -> None:
    script = f"""
import bpy
from pathlib import Path

obj_path = Path({str(obj_path)!r})
blend_path = Path({str(blend_path)!r})
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
if hasattr(bpy.ops.wm, 'obj_import'):
    bpy.ops.wm.obj_import(filepath=str(obj_path))
else:
    bpy.ops.import_scene.obj(filepath=str(obj_path))
for obj in bpy.context.scene.objects:
    obj.select_set(obj.type == 'MESH')
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
blend_path.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
"""
    script_path = blend_path.with_name(".obj_to_blend.py")
    script_path.write_text(script, encoding="utf-8")
    try:
        subprocess.run(
            [blender_path, "--background", "--python", str(script_path)],
            check=True,
        )
    finally:
        script_path.unlink(missing_ok=True)


def resolve_instantmesh_checkpoint(config_path: str) -> Path:
    model_ckpt_path = Path(config_path)
    if model_ckpt_path.exists():
        return model_ckpt_path

    candidates = []
    cache_roots = [
        Path.cwd() / "ckpts",
        Path("D:/hf_cache"),
        Path("D:/InstantMesh2/outputs/hf_cache"),
    ]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home))
    filename = model_ckpt_path.name
    for root in cache_roots:
        if not root.exists():
            continue
        candidates.extend(root.rglob(filename))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"InstantMesh checkpoint not found: {model_ckpt_path}")


def main() -> None:
    args = parse_args()
    study_dir = Path(args.study_dir).resolve()
    manifest = load_manifest(study_dir)
    jobs = list(iter_result_jobs(study_dir, manifest, args.only_object, args.only_result))
    if not jobs:
        raise ValueError("No user-study reconstruction jobs matched the filters.")

    config = OmegaConf.load(args.config)
    config_name = Path(args.config).stem
    model_config = config.model_config
    infer_config = config.infer_config
    if args.mesh_grid_res is not None and hasattr(model_config, "params") and "grid_res" in model_config.params:
        model_config.params.grid_res = args.mesh_grid_res
    if args.render_resolution is not None:
        infer_config.render_resolution = args.render_resolution
    if args.texture_resolution is not None:
        infer_config.texture_resolution = args.texture_resolution
    is_flexicubes = config_name.startswith("instant-mesh")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("InstantMesh reconstruction requires CUDA for this workflow.")

    print(f"[user-study] jobs={len(jobs)}")
    print("[user-study] loading reconstruction model")
    model = instantiate_from_config(model_config)
    model_ckpt_path = resolve_instantmesh_checkpoint(str(infer_config.model_path))
    print(f"[user-study] checkpoint={model_ckpt_path}")
    state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    if is_flexicubes:
        model.init_flexicubes_geometry(device, fovy=30.0)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * args.scale).to(device)
    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5], dtype=torch.long, device=device)
        input_cameras = input_cameras[:, indices]
    else:
        indices = None

    completed = []
    failed = []
    for idx, job in enumerate(jobs, start=1):
        model_dir = job["model_dir"]
        obj_path = model_dir / "result.obj"
        blend_path = model_dir / "result.blend"
        if args.skip_existing and obj_path.exists() and blend_path.exists():
            print(f"[user-study] [{idx}/{len(jobs)}] skip existing {job['name']}")
            completed.append(job)
            continue

        print(f"[user-study] [{idx}/{len(jobs)}] reconstruct {job['name']}")
        model_dir.mkdir(parents=True, exist_ok=True)
        running_marker = model_dir / "RUNNING.txt"
        running_marker.write_text(str(job["sheet_path"]), encoding="utf-8")
        try:
            images = load_sheet(job["sheet_path"]).unsqueeze(0).to(device)
            if indices is not None:
                images = images[:, indices]
            images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

            with torch.no_grad():
                planes = model.forward_planes(images, input_cameras)
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
                    str(obj_path),
                )
            else:
                vertices, faces, vertex_colors = mesh_out
                save_obj(vertices, faces, vertex_colors, str(obj_path))
            del images, planes, mesh_out
            torch.cuda.empty_cache()

            obj_to_blend(args.blender_path, obj_path, blend_path)
            (model_dir / "DONE.json").unlink(missing_ok=True)
            running_marker.unlink(missing_ok=True)
            completed.append(job)
        except Exception as exc:
            failed.append({"job": job, "error": str(exc)})
            (model_dir / "FAILED.txt").write_text(str(exc), encoding="utf-8")
            running_marker.unlink(missing_ok=True)
            print(f"[user-study] FAILED {job['name']}: {exc}", file=sys.stderr)
            break

    summary = {
        "completed": len(completed),
        "failed": failed,
        "total": len(jobs),
    }
    (study_dir / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if failed:
        raise RuntimeError(f"Stopped after failure: {failed[0]['job']['name']}: {failed[0]['error']}")
    print(f"[user-study] completed={len(completed)}/{len(jobs)}")


if __name__ == "__main__":
    main()
