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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_pose_list(text: str) -> list[tuple[float, float]]:
    poses = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Pose must look like azimuth:elevation, got {item!r}")
        azimuth, elevation = item.split(":", 1)
        poses.append((float(azimuth) % 360.0, float(elevation)))
    if len(poses) != 3:
        raise ValueError(f"Expected exactly 3 reference poses, got {len(poses)}.")
    return poses


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one input image and three pose-metadata reference views from a single 3D model."
    )
    parser.add_argument("--model_path", required=True, help="Path to one .glb, .gltf, .obj, or .fbx file.")
    parser.add_argument("--output_dir", default="outputs/reference_quick_test_views")
    parser.add_argument("--blender_path", default="blender")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--render_size", type=int, default=768)
    parser.add_argument("--exposure", type=float, default=-1.0)
    parser.add_argument("--postprocess_python", default=sys.executable)
    parser.add_argument("--input_azimuth", type=float, default=0.0)
    parser.add_argument("--input_elevation", type=float, default=0.0)
    parser.add_argument(
        "--ref_poses",
        default="90:-10,150:20,270:20",
        help="Exactly three comma-separated azimuth:elevation pairs. Default matches useful v1.2 side/back-ish refs.",
    )
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--fov", type=float, default=30.0)
    parser.add_argument("--camera_padding", type=float, default=0.86)
    parser.add_argument("--target_size", type=float, default=1.6)
    parser.add_argument("--orientation_metadata", default=None)
    parser.add_argument("--default_yaw_offset", type=float, default=0.0)
    parser.add_argument("--force_opaque", action="store_true")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    elif bpy is None:
        argv = argv[1:]
    else:
        argv = []
    args = parser.parse_args(argv)
    args.ref_pose_values = parse_pose_list(args.ref_poses)
    return args


def run_blender_wrapper(args):
    script_path = Path(__file__).resolve()
    cmd = [
        args.blender_path,
        "--background",
        "--python",
        str(script_path),
        "--",
        "--model_path",
        args.model_path,
        "--output_dir",
        args.output_dir,
        "--image_size",
        str(args.image_size),
        "--render_size",
        str(args.render_size),
        "--exposure",
        str(args.exposure),
        "--postprocess_python",
        args.postprocess_python,
        "--input_azimuth",
        str(args.input_azimuth),
        "--input_elevation",
        str(args.input_elevation),
        "--ref_poses",
        args.ref_poses,
        "--radius",
        str(args.radius),
        "--fov",
        str(args.fov),
        "--camera_padding",
        str(args.camera_padding),
        "--target_size",
        str(args.target_size),
        "--default_yaw_offset",
        str(args.default_yaw_offset),
    ]
    if args.orientation_metadata:
        cmd.extend(["--orientation_metadata", args.orientation_metadata])
    if args.force_opaque:
        cmd.append("--force_opaque")
    print("[quick-render] launching Blender:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    subprocess.run(cmd, check=True)


def ref_filename(azimuth: float, elevation: float) -> str:
    az = int(round(azimuth)) % 360
    el = int(round(elevation))
    sign = "" if el < 0 else ""
    return f"ref_{az:03d}_el{sign}{el}.png"


def main():
    args = parse_args()
    if bpy is None:
        run_blender_wrapper(args)
        return

    from scripts.render_reference_training_dataset import (
        clear_scene,
        compute_camera_radius,
        force_materials_opaque,
        import_model,
        load_orientation_metadata,
        normalize_scene,
        remove_imported_non_meshes,
        remove_low_detail_outlier_roots,
        remove_named_meshes,
        render_view,
        resolve_model_render_options,
        setup_render,
        write_meta,
    )

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    import_model(model_path)
    remove_imported_non_meshes()

    orientation_metadata = load_orientation_metadata(args.orientation_metadata)
    render_options = resolve_model_render_options(
        model_path,
        orientation_metadata,
        default_yaw_offset=args.default_yaw_offset,
    )
    if render_options.get("remove_meshes"):
        remove_named_meshes(render_options["remove_meshes"])
    if args.force_opaque or render_options.get("force_opaque"):
        force_materials_opaque()
    removed_helpers = remove_low_detail_outlier_roots()

    norm = normalize_scene(args.target_size, yaw_offset_degrees=render_options["yaw_degrees"])
    camera_radius = compute_camera_radius(norm, args.fov, args.radius, args.camera_padding)
    camera = setup_render(args.render_size, args.fov, args.exposure)

    rendered_views = {}
    quality = {}
    input_report = render_view(
        camera,
        output_dir / "input.png",
        args.input_azimuth,
        args.input_elevation,
        camera_radius,
        norm["look_at"],
        args,
    )
    rendered_views["input.png"] = {
        "azimuth": args.input_azimuth,
        "elevation": args.input_elevation,
    }
    quality["input.png"] = input_report

    refs_dir = output_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    ref_metadata = {}
    for azimuth, elevation in args.ref_pose_values:
        filename = ref_filename(azimuth, elevation)
        report = render_view(
            camera,
            refs_dir / filename,
            azimuth,
            elevation,
            camera_radius,
            norm["look_at"],
            args,
        )
        rendered_views[filename] = {"azimuth": azimuth, "elevation": elevation}
        ref_metadata[filename] = {"azimuth": azimuth, "elevation": elevation}
        quality[filename] = report

    meta = {
        "source_model": str(model_path),
        "output_dir": str(output_dir),
        "image_size": args.image_size,
        "render_size": args.render_size,
        "exposure": args.exposure,
        "input_image": "input.png",
        "input_azimuth": args.input_azimuth,
        "input_elevation": args.input_elevation,
        "reference_dir": "refs",
        "reference_images": [f"refs/{filename}" for filename in ref_metadata],
        "ref_metadata": ref_metadata,
        "rendered_views": rendered_views,
        "render_options": render_options,
        "removed_helper_roots": removed_helpers,
        "camera_radius": camera_radius,
        **norm,
    }
    write_meta(output_dir / "meta.json", meta)
    write_meta(refs_dir / "ref_metadata.json", ref_metadata)
    write_meta(output_dir / "quality_report.json", quality)
    print(f"[quick-render] wrote {output_dir / 'input.png'}")
    print(f"[quick-render] wrote {refs_dir / 'ref_metadata.json'}")
    print("[quick-render] references:")
    for filename, pose in ref_metadata.items():
        print(f"  {filename}: azimuth={pose['azimuth']} elevation={pose['elevation']}")


if __name__ == "__main__":
    main()
