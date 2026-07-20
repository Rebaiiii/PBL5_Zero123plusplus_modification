from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SLOT_POSES = {
    "v1.1": [(30, 30), (90, -20), (150, 30), (210, -20), (270, 30), (330, -20)],
    "v1.2": [(30, 20), (90, -10), (150, 20), (210, -10), (270, 20), (330, -10)],
}
SUPPORTED_EXTS = (".glb", ".gltf", ".obj", ".fbx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render six Zero123++ target-slot images from user-study source object models."
    )
    parser.add_argument("--study_dir", default="outputs/user_study")
    parser.add_argument("--source_views_dir", default="outputs/rag_quick_test_views")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--blender_path", default=r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe")
    parser.add_argument("--postprocess_python", default=sys.executable)
    parser.add_argument("--zero123plus_pose_version", choices=sorted(SLOT_POSES), default="v1.2")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--render_size", type=int, default=768)
    parser.add_argument("--exposure", type=float, default=-1.0)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--fov", type=float, default=30.0)
    parser.add_argument("--camera_padding", type=float, default=0.86)
    parser.add_argument("--target_size", type=float, default=1.6)
    parser.add_argument("--only_object", default=None)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    elif bpy is None:
        argv = argv[1:]
    else:
        argv = []
    return parser.parse_args(argv)


def run_blender_wrapper(args: argparse.Namespace) -> None:
    cmd = [
        args.blender_path,
        "--background",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--study_dir",
        args.study_dir,
        "--source_views_dir",
        args.source_views_dir,
        "--blender_path",
        args.blender_path,
        "--postprocess_python",
        args.postprocess_python,
        "--zero123plus_pose_version",
        args.zero123plus_pose_version,
        "--image_size",
        str(args.image_size),
        "--render_size",
        str(args.render_size),
        "--exposure",
        str(args.exposure),
        "--radius",
        str(args.radius),
        "--fov",
        str(args.fov),
        "--camera_padding",
        str(args.camera_padding),
        "--target_size",
        str(args.target_size),
    ]
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    if args.only_object:
        cmd.extend(["--only_object", args.only_object])
    if args.skip_existing:
        cmd.append("--skip_existing")
    else:
        cmd.append("--no-skip_existing")
    print("[gt-render] launching Blender:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_manifest(study_dir: Path) -> list[dict]:
    manifest = load_json(study_dir / "study_manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("objects"), list):
        raise FileNotFoundError(f"Study manifest not found or invalid: {study_dir / 'study_manifest.json'}")
    return [entry for entry in manifest["objects"] if isinstance(entry, dict) and entry.get("object_id")]


def candidate_source_model_paths(object_id: str, study_dir: Path, source_views_dir: Path) -> list[Path]:
    names = [object_id, object_id.replace("-", "_"), object_id.replace("_", "-")]
    candidates: list[Path] = []
    meta = load_json(source_views_dir / object_id / "meta.json")
    if isinstance(meta, dict) and meta.get("source_model"):
        candidates.append(Path(meta["source_model"]))
    for root in (
        study_dir / "source_models",
        REPO_ROOT,
        REPO_ROOT / "test_objects",
        REPO_ROOT / "data" / "source_models",
        REPO_ROOT / "data" / "objaverse_toy_models",
    ):
        for name in names:
            for ext in SUPPORTED_EXTS:
                candidates.append(root / f"{name}{ext}")
    if object_id == "dog":
        for name in ("dog", "animal_toy", "orange_dog", "dog_toy"):
            for root in (study_dir / "source_models", REPO_ROOT, REPO_ROOT / "test_objects", REPO_ROOT / "data" / "source_models"):
                for ext in SUPPORTED_EXTS:
                    candidates.append(root / f"{name}{ext}")
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in seen:
            existing.append(resolved)
            seen.add(resolved)
    return existing


def render_object(
    object_id: str,
    source_model: Path,
    output_dir: Path,
    source_meta: dict | None,
    args: argparse.Namespace,
) -> dict:
    from scripts.render_rag_zero123plus_tiny import (
        clear_scene,
        compute_camera_radius,
        force_materials_opaque,
        import_model,
        normalize_scene,
        remove_imported_non_meshes,
        remove_low_detail_outlier_roots,
        remove_named_meshes,
        render_view,
        resolve_model_render_options,
        setup_render,
        write_meta,
    )

    clear_scene()
    import_model(source_model)
    remove_imported_non_meshes()

    render_options = {}
    if isinstance(source_meta, dict) and isinstance(source_meta.get("render_options"), dict):
        render_options = dict(source_meta["render_options"])
    else:
        render_options = resolve_model_render_options(source_model, {})
    if render_options.get("remove_meshes"):
        remove_named_meshes(render_options["remove_meshes"])
    if render_options.get("force_opaque"):
        force_materials_opaque()
    removed_helpers = remove_low_detail_outlier_roots()

    target_size = args.target_size
    if isinstance(source_meta, dict):
        if source_meta.get("target_size") is not None:
            target_size = float(source_meta["target_size"])
        elif isinstance(source_meta.get("bbox_size_after"), list) and source_meta["bbox_size_after"]:
            target_size = max(float(value) for value in source_meta["bbox_size_after"])
    norm = normalize_scene(target_size, yaw_offset_degrees=float(render_options.get("yaw_degrees", 0.0)))
    fov = float(source_meta.get("fov", args.fov)) if isinstance(source_meta, dict) else args.fov
    render_size = int(source_meta.get("render_size", args.render_size)) if isinstance(source_meta, dict) else args.render_size
    image_size = int(source_meta.get("image_size", args.image_size)) if isinstance(source_meta, dict) else args.image_size
    exposure = float(source_meta.get("exposure", args.exposure)) if isinstance(source_meta, dict) else args.exposure
    camera_radius = (
        float(source_meta["camera_radius"])
        if isinstance(source_meta, dict) and source_meta.get("camera_radius") is not None
        else compute_camera_radius(norm, fov, args.radius, args.camera_padding)
    )

    camera = setup_render(render_size, fov, exposure)
    output_dir.mkdir(parents=True, exist_ok=True)

    render_args = argparse.Namespace(
        image_size=image_size,
        postprocess_python=args.postprocess_python,
    )
    rendered_views = {}
    quality = {}
    for azimuth, elevation in SLOT_POSES[args.zero123plus_pose_version]:
        filename = f"target_{azimuth:03d}.png"
        report = render_view(
            camera,
            output_dir / filename,
            float(azimuth),
            float(elevation),
            camera_radius,
            norm["look_at"],
            render_args,
        )
        rendered_views[filename] = {"azimuth": float(azimuth), "elevation": float(elevation)}
        quality[filename] = report

    meta = {
        "object_id": object_id,
        "source_model": str(source_model),
        "source_meta": source_meta or {},
        "output_dir": str(output_dir),
        "zero123plus_pose_version": args.zero123plus_pose_version,
        "target_images": list(rendered_views),
        "rendered_views": rendered_views,
        "quality": quality,
        "render_options": render_options,
        "removed_helper_roots": removed_helpers,
        "camera_radius": camera_radius,
        **norm,
    }
    write_meta(output_dir / "meta.json", meta)
    write_meta(output_dir / "quality_report.json", quality)
    return meta


def main() -> int:
    args = parse_args()
    if bpy is None:
        run_blender_wrapper(args)
        return 0

    study_dir = Path(args.study_dir).resolve()
    source_views_dir = Path(args.source_views_dir).resolve()
    output_root = Path(args.output_dir).resolve() if args.output_dir else study_dir / "ground_truth_targets"
    objects = load_manifest(study_dir)
    rendered = []
    skipped = []
    for entry in objects:
        object_id = entry["object_id"]
        if args.only_object and object_id != args.only_object:
            continue
        object_output = output_root / object_id
        expected = [object_output / f"target_{azimuth:03d}.png" for azimuth, _ in SLOT_POSES[args.zero123plus_pose_version]]
        if args.skip_existing and all(path.exists() for path in expected):
            print(f"[gt-render] skip existing {object_id}: {object_output}")
            skipped.append({"object_id": object_id, "reason": "existing"})
            continue
        source_models = candidate_source_model_paths(object_id, study_dir, source_views_dir)
        if not source_models:
            print(f"[gt-render] missing source model for {object_id}")
            skipped.append({"object_id": object_id, "reason": "missing_source_model"})
            continue
        source_meta = load_json(source_views_dir / object_id / "meta.json") or {}
        meta = render_object(object_id, source_models[0], object_output, source_meta, args)
        rendered.append({"object_id": object_id, "output_dir": str(object_output), "source_model": meta["source_model"]})

    summary = {"rendered": rendered, "skipped": skipped}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[gt-render] wrote {output_root / 'render_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
