from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Condition:
    name: str
    adapter_ckpt: Optional[str] = None
    refs_path: Optional[str] = None
    metadata_path: Optional[str] = None
    token_scale: float = 0.1
    global_scale: float = 0.05
    spatial_gate_scale: float = 1.0
    spatial_gating: bool = True
    shuffle_metadata: bool = False
    zero_reference_influence: bool = False


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def extract_rag_adapter_state_dict(checkpoint):
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("RAG adapter checkpoint must be a state_dict or contain a 'state_dict' entry.")
    prefixes = ("rag_adapter.", "model.rag_adapter.")
    adapter_state = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                adapter_state[key[len(prefix):]] = value
                break
    if adapter_state:
        return adapter_state
    if any(key.startswith(("ref_proj.", "view_embed.")) for key in state_dict):
        return state_dict
    raise ValueError("No RAG adapter weights found in checkpoint.")


def load_adapter(checkpoint_path: str, embed_dim: int, device: torch.device):
    import torch

    from zero123plus.rag_adapter import RAGReferenceAdapter

    adapter = RAGReferenceAdapter(embed_dim=embed_dim).to(device=device, dtype=torch.float16)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    adapter_state = extract_rag_adapter_state_dict(checkpoint)
    incompatible = adapter.load_state_dict(adapter_state, strict=False)
    adapter.eval()
    return adapter, list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def adapter_weight_norms(adapter):
    ref_proj_sq = 0.0
    view_embed_sq = 0.0
    for name, param in adapter.named_parameters():
        value = float(param.detach().float().pow(2).sum().item())
        if name.startswith("ref_proj"):
            ref_proj_sq += value
        elif name.startswith("view_embed"):
            view_embed_sq += value
    return ref_proj_sq ** 0.5, view_embed_sq ** 0.5


def load_input_image(path: str, no_rembg: bool):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Input image not found: {Path(path).resolve()}\n"
            "Use an existing image path, or put the file in the expected folder."
        )
    image = Image.open(path)
    if no_rembg:
        return image.convert("RGB")
    import rembg
    from src.utils.infer_util import remove_background, resize_foreground

    session = rembg.new_session()
    image = remove_background(image, session)
    image = resize_foreground(image, 0.85)
    return image.convert("RGB")


def load_pipeline(config_path: str, device: torch.device):
    import torch
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    infer_config = config.infer_config
    print("Loading diffusion model ...")
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing",
    )
    print("Loading custom white-background unet ...")
    if os.path.exists(infer_config.unet_path):
        unet_ckpt_path = infer_config.unet_path
    else:
        unet_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="diffusion_pytorch_model.bin",
            repo_type="model",
        )
    state_dict = torch.load(unet_ckpt_path, map_location="cpu")
    pipeline.unet.load_state_dict(state_dict, strict=True)
    return pipeline.to(device)


def shuffled_slot_weights(slot_weights: List[List[float]]) -> List[List[float]]:
    if len(slot_weights) <= 1:
        return [list(reversed(weights)) for weights in slot_weights]
    return slot_weights[1:] + slot_weights[:1]


def split_zero123plus_sheet(image: Image.Image) -> List[Image.Image]:
    width, height = image.size
    tile_w = width // 2
    tile_h = height // 3
    tiles = []
    for row in range(3):
        for col in range(2):
            tiles.append(image.crop((col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h)))
    return tiles


def make_labeled_grid(rows: Iterable[tuple[str, Image.Image]], output_path: Path):
    rows = list(rows)
    if not rows:
        return
    font = ImageFont.load_default()
    label_w = 230
    row_h = max(image.height for _, image in rows)
    image_w = max(image.width for _, image in rows)
    grid = Image.new("RGB", (label_w + image_w, row_h * len(rows)), "white")
    draw = ImageDraw.Draw(grid)
    for row_idx, (label, image) in enumerate(rows):
        y = row_idx * row_h
        draw.text((10, y + 10), label, fill=(0, 0, 0), font=font)
        grid.paste(image.convert("RGB"), (label_w, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def save_condition_config(path: Path, condition: Condition, extra: dict):
    payload = {
        "condition": condition.__dict__,
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_metrics(condition_images: dict[str, Image.Image], pose_version: str):
    from zero123plus.rag_utils import get_zero123plus_target_poses

    if "baseline_no_adapter" not in condition_images:
        return {}, []

    baseline = condition_images["baseline_no_adapter"]
    baseline_tiles = split_zero123plus_sheet(baseline)
    target_azimuths, target_elevations = get_zero123plus_target_poses(pose_version)
    rows = []
    metrics = {}
    for condition_name, image in condition_images.items():
        tiles = split_zero123plus_sheet(image)
        slot_values = []
        for slot, (baseline_tile, tile) in enumerate(zip(baseline_tiles, tiles)):
            base = np.asarray(baseline_tile, dtype=np.float32) / 255.0
            value = np.asarray(tile, dtype=np.float32) / 255.0
            mad = float(np.abs(value - base).mean())
            slot_values.append(mad)
            rows.append({
                "condition": condition_name,
                "slot": slot,
                "azimuth": target_azimuths[slot],
                "elevation": target_elevations[slot],
                "mean_abs_diff_from_baseline": mad,
            })
        metrics[condition_name] = {
            "mean_abs_diff_from_baseline": float(np.mean(slot_values)),
            "front_slots_mean_abs_diff": float(np.mean([slot_values[0], slot_values[5]])),
            "side_back_slots_mean_abs_diff": float(np.mean(slot_values[1:5])),
            "slots": slot_values,
        }
    return metrics, rows


def run_condition(
    condition: Condition,
    condition_index: int,
    condition_total: int,
    pipeline,
    adapters,
    input_image,
    args,
    device,
    output_dir: Path,
):
    from src.utils.rag_rerank import generate_zero123plus_candidate
    from zero123plus.rag_utils import (
        load_reference_images,
        references_to_pil,
        references_to_slot_weights,
        references_to_view_ids,
    )

    condition_dir = output_dir / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    log_path = condition_dir / "debug.log"
    with open(log_path, "w", encoding="utf-8") as log_file, contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
        print(f"[ABLATION] condition {condition_index}/{condition_total}: {condition.name}", flush=True)
        print(f"[ABLATION] diffusion_steps = {args.diffusion_steps}", flush=True)
        print("[ABLATION] Zero123++ sheet generation only; InstantMesh reconstruction is not running.", flush=True)
        print("[RAG-ADAPTER] auto_view_assignment = disabled", flush=True)
        print(f"[RAG-ADAPTER] spatial_gating = {str(condition.spatial_gating).lower()}", flush=True)
        print(f"[RAG-ADAPTER] rag_token_scale = {condition.token_scale}", flush=True)
        print(f"[RAG-ADAPTER] rag_global_scale = {condition.global_scale}", flush=True)
        print(f"[RAG-ADAPTER] rag_spatial_gate_scale = {condition.spatial_gate_scale}", flush=True)
        references = []
        ref_slot_weights = None
        ref_view_ids = None
        adapter = None
        checkpoint_loaded = False
        missing_keys = []
        unexpected_keys = []

        if condition.adapter_ckpt:
            adapter, missing_keys, unexpected_keys = adapters[condition.adapter_ckpt]
            checkpoint_loaded = True
            ref_proj_norm, view_embed_norm = adapter_weight_norms(adapter)
            print(f"[RAG-ADAPTER] checkpoint_loaded = true")
            print(f"[RAG-ADAPTER] checkpoint_path = {condition.adapter_ckpt}")
            print(f"[RAG-ADAPTER] missing_keys = {missing_keys}")
            print(f"[RAG-ADAPTER] unexpected_keys = {unexpected_keys}")
            print(f"[RAG-ADAPTER] ref_proj weight norm = {ref_proj_norm:.6f}")
            print(f"[RAG-ADAPTER] view_embed weight norm = {view_embed_norm:.6f}")
        else:
            print("[ABLATION] baseline/no adapter path: no checkpoint, no references, normal Zero123++.", flush=True)
            print("[RAG-ADAPTER] checkpoint_loaded = false", flush=True)

        if condition.refs_path and not condition.zero_reference_influence:
            references = load_reference_images(
                condition.refs_path,
                metadata_path=condition.metadata_path,
                pose_version=args.zero123plus_pose_version,
                max_image_size=args.rag_max_size,
            )
            ref_slot_weights = references_to_slot_weights(
                references,
                pose_version=args.zero123plus_pose_version,
                azimuth_sigma=args.rag_ref_azimuth_sigma,
                elevation_sigma=args.rag_ref_elevation_sigma,
            )
            if condition.shuffle_metadata:
                ref_slot_weights = shuffled_slot_weights(ref_slot_weights)
            ref_view_ids = references_to_view_ids(references)
        elif condition.zero_reference_influence:
            print("[RAG-ADAPTER] references intentionally disabled for no_refs condition")

        known = sum(ref.azimuth is not None and ref.elevation is not None for ref in references)
        print(f"[RAG-ADAPTER] references_known = {known}/{len(references)}", flush=True)
        if references:
            for ref in references:
                print(
                    "[RAG-ADAPTER] "
                    f"reference={ref.path} view_label={ref.view_label} view_id={ref.view_id} "
                    f"azimuth={ref.azimuth} elevation={ref.elevation} pose_source={ref.pose_source}"
                )
        if ref_slot_weights is not None:
            print(f"[RAG-ADAPTER] ref_slot_weights = {ref_slot_weights}", flush=True)
        else:
            print("[RAG-ADAPTER] ref_slot_weights = null", flush=True)

        save_condition_config(
            condition_dir / "condition_config.json",
            condition,
            {
                "seed": args.seed,
                "diffusion_steps": args.diffusion_steps,
                "checkpoint_loaded": checkpoint_loaded,
                "references_known": f"{known}/{len(references)}",
                "auto_view_assignment": "disabled",
                "ref_slot_weights": ref_slot_weights,
            },
        )
        (condition_dir / "ref_slot_weights.json").write_text(
            json.dumps(ref_slot_weights, indent=2),
            encoding="utf-8",
        )

        running_path = condition_dir / "RUNNING.txt"
        running_path.write_text(
            f"Generating {condition.name} with seed={args.seed}, steps={args.diffusion_steps}\n",
            encoding="utf-8",
        )
        print(
            f"[ABLATION] generating Zero123++ sheet for {condition.name}; "
            "this is the slow part.",
            flush=True,
        )
        output_image = generate_zero123plus_candidate(
            pipeline,
            input_image,
            args.diffusion_steps,
            device,
            args.seed,
            rag_reference_images=references_to_pil(references) if references else None,
            rag_ref_view_ids=ref_view_ids,
            rag_ref_slot_weights=ref_slot_weights,
            rag_adapter=adapter,
            rag_token_scale=condition.token_scale,
            rag_global_scale=condition.global_scale,
            rag_spatial_gating=condition.spatial_gating,
            rag_spatial_gate_scale=condition.spatial_gate_scale,
            rag_debug_dump=args.rag_debug_dump,
        )
        output_path = condition_dir / "zero123plus_sheet.png"
        output_image.save(output_path)
        running_path.unlink(missing_ok=True)
        print(f"[ABLATION] saved {output_path}", flush=True)
        return output_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--rag_refs", required=True)
    parser.add_argument("--rag_ref_metadata", default=None)
    parser.add_argument("--wrong_rag_refs", default=None)
    parser.add_argument("--wrong_rag_ref_metadata", default=None)
    parser.add_argument("--adapter_last", required=True)
    parser.add_argument("--adapter_step500", default=None)
    parser.add_argument("--output_dir", default="outputs/rag_ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion_steps", type=int, default=75)
    parser.add_argument("--zero123plus_pose_version", choices=["v1.1", "v1.2"], default="v1.2")
    parser.add_argument("--rag_max_size", type=int, default=1024)
    parser.add_argument("--rag_ref_azimuth_sigma", type=float, default=45.0)
    parser.add_argument("--rag_ref_elevation_sigma", type=float, default=25.0)
    parser.add_argument("--no_rembg", action="store_true")
    parser.add_argument("--rag_debug_dump", action="store_true")
    parser.add_argument("--run_reconstruction", action="store_true")
    parser.add_argument("--skip_slot_grids", action="store_true", help="Only write full-sheet comparison outputs, not per-slot grids.")
    parser.add_argument(
        "--conditions",
        default=None,
        help=(
            "Comma-separated condition names. Supported: baseline_no_adapter, "
            "adapter_no_refs, adapter_correct_refs_normal, adapter_correct_refs_strong, "
            "adapter_step500_normal_scale, adapter_step500_strong_scale, "
            "adapter_last_shuffled_metadata, adapter_last_wrong_refs. "
            "Legacy names adapter_last_no_refs/normal_scale/strong_scale also work."
        ),
    )
    return parser.parse_args()


def validate_existing_path(path: Optional[str], label: str, required: bool = True):
    if not path:
        if required:
            raise ValueError(f"Missing required {label}.")
        return
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {Path(path).resolve()}")


def main():
    import torch
    from pytorch_lightning import seed_everything
    from zero123plus.rag_utils import get_zero123plus_target_poses

    args = parse_args()
    if args.run_reconstruction:
        raise NotImplementedError(
            "This ablation script focuses on Zero123++ sheets. Run InstantMesh reconstruction separately after selecting a condition."
        )
    validate_existing_path(args.config, "--config")
    validate_existing_path(args.input, "--input")
    validate_existing_path(args.rag_refs, "--rag_refs")
    validate_existing_path(args.rag_ref_metadata, "--rag_ref_metadata", required=False)
    validate_existing_path(args.wrong_rag_refs, "--wrong_rag_refs", required=False)
    validate_existing_path(args.wrong_rag_ref_metadata, "--wrong_rag_ref_metadata", required=False)
    validate_existing_path(args.adapter_last, "--adapter_last")
    requested_names = [item.strip() for item in args.conditions.split(",")] if args.conditions else None
    needs_step500 = requested_names is None or any("step500" in name for name in requested_names)
    validate_existing_path(args.adapter_step500, "--adapter_step500", required=needs_step500)

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image = load_input_image(args.input, args.no_rembg)
    input_image.save(output_dir / "input_image.png")
    pipeline = load_pipeline(args.config, device)
    embed_dim = getattr(pipeline.vision_encoder.config, "projection_dim", None)
    if embed_dim is None:
        embed_dim = getattr(pipeline.vision_encoder.config, "hidden_size", None)
    if embed_dim is None:
        raise ValueError("Could not infer CLIP vision embedding dimension for RAG adapter.")

    checkpoint_paths = [args.adapter_last]
    if args.adapter_step500:
        checkpoint_paths.append(args.adapter_step500)
    adapters = {}
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path not in adapters:
            print(f"[ABLATION] loading adapter checkpoint {checkpoint_path}")
            adapters[checkpoint_path] = load_adapter(checkpoint_path, embed_dim, device)

    normal = {"token_scale": 0.1, "global_scale": 0.05, "spatial_gate_scale": 1.0}
    strong = {"token_scale": 0.5, "global_scale": 0.0, "spatial_gate_scale": 5.0}
    condition_map = {
        "baseline_no_adapter": Condition("baseline_no_adapter", spatial_gating=False),
        "adapter_no_refs": Condition("adapter_no_refs", args.adapter_last, None, None, spatial_gating=False, zero_reference_influence=True),
        "adapter_last_no_refs": Condition("adapter_no_refs", args.adapter_last, None, None, spatial_gating=False, zero_reference_influence=True),
        "adapter_correct_refs_normal": Condition("adapter_correct_refs_normal", args.adapter_last, args.rag_refs, args.rag_ref_metadata, **normal),
        "adapter_last_normal_scale": Condition("adapter_correct_refs_normal", args.adapter_last, args.rag_refs, args.rag_ref_metadata, **normal),
        "adapter_correct_refs_strong": Condition("adapter_correct_refs_strong", args.adapter_last, args.rag_refs, args.rag_ref_metadata, **strong),
        "adapter_last_strong_scale": Condition("adapter_correct_refs_strong", args.adapter_last, args.rag_refs, args.rag_ref_metadata, **strong),
        "adapter_last_shuffled_metadata": Condition("adapter_last_shuffled_metadata", args.adapter_last, args.rag_refs, args.rag_ref_metadata, **normal, shuffle_metadata=True),
    }
    if args.adapter_step500:
        condition_map["adapter_step500_normal_scale"] = Condition(
            "adapter_step500_normal_scale", args.adapter_step500, args.rag_refs, args.rag_ref_metadata, **normal
        )
        condition_map["adapter_step500_strong_scale"] = Condition(
            "adapter_step500_strong_scale", args.adapter_step500, args.rag_refs, args.rag_ref_metadata, **strong
        )
    if args.wrong_rag_refs:
        condition_map["adapter_last_wrong_refs"] = Condition(
            "adapter_last_wrong_refs",
            args.adapter_last,
            args.wrong_rag_refs,
            args.wrong_rag_ref_metadata,
            **normal,
        )
    default_conditions = [
        "baseline_no_adapter",
        "adapter_correct_refs_normal",
        "adapter_correct_refs_strong",
        "adapter_step500_normal_scale",
        "adapter_step500_strong_scale",
        "adapter_no_refs",
        "adapter_last_shuffled_metadata",
    ]
    requested_names = requested_names or default_conditions
    missing_conditions = [name for name in requested_names if name not in condition_map]
    if missing_conditions:
        raise ValueError(f"Unknown or unavailable condition(s): {missing_conditions}")
    conditions = [condition_map[name] for name in requested_names]

    condition_images = {}
    for condition_index, condition in enumerate(conditions, start=1):
        seed_everything(args.seed)
        torch.cuda.empty_cache()
        condition_images[condition.name] = run_condition(
            condition,
            condition_index,
            len(conditions),
            pipeline,
            adapters,
            input_image,
            args,
            device,
            output_dir,
        )

    make_labeled_grid(condition_images.items(), output_dir / "comparison_grid.png")
    if not args.skip_slot_grids:
        target_azimuths, target_elevations = get_zero123plus_target_poses(args.zero123plus_pose_version)
        for slot, (azimuth, elevation) in enumerate(zip(target_azimuths, target_elevations)):
            slot_rows = []
            for condition_name, image in condition_images.items():
                slot_rows.append((condition_name, split_zero123plus_sheet(image)[slot]))
            make_labeled_grid(
                slot_rows,
                output_dir / f"slot_{slot}_az{azimuth}_el{elevation}.png",
            )

    metrics, metric_rows = compute_metrics(condition_images, args.zero123plus_pose_version)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if metric_rows:
        with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)
    else:
        (output_dir / "metrics.csv").write_text("", encoding="utf-8")

    print("[ABLATION] comparison_grid:", output_dir / "comparison_grid.png")
    print("[ABLATION] If correct refs differ from no_refs, adapter is using references.")
    print("[ABLATION] If strong scale changes output more than normal scale, adapter influence exists.")
    print("[ABLATION] If shuffled metadata changes the wrong tiles, spatial routing is active.")
    print("[ABLATION] If everything is identical to baseline, adapter influence is too weak or not applied.")


if __name__ == "__main__":
    main()
