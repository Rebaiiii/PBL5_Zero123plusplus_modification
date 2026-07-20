from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None
    Vector = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_objaverse_rag_dataset as builder


REMOVE_IDS = {
    "object_000059",
    "object_000097",
    "object_000102",
    "object_000104",
    "object_000147",
    "object_000163",
    "object_000180",
    "object_000210",
    "object_000348",
    "object_000376",
    "object_000426",
    "object_000469",
}

YAW_DEGREES = {
    "object_000094": 90.0,
    "object_000096": 90.0,
    "object_000130": 180.0,
    "object_000343": 180.0,
    "object_000355": 180.0,
    "object_000365": 90.0,
    "object_000415": 180.0,
    "object_000462": 90.0,
    "object_000467": 180.0,
    "object_000476": 180.0,
    "object_000498": 180.0,
    "object_000501": 180.0,
}

REMOVE_GROUND_IDS = {"object_000116", "object_000440"}
REMOVE_FAR_IDS = {"object_000462"}


def parse_object_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "object" and parts[1].isdigit():
            ids.append(f"object_{int(parts[1]):06d}")
    return ids


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def object_dir(dataset_root: Path, object_id: str) -> Path:
    return dataset_root / "objects" / object_id


def remove_records(dataset_root: Path, remove_ids: set[str]) -> None:
    for filename in ("train.jsonl", "val.jsonl", "test_objaverse_heldout.jsonl"):
        path = dataset_root / filename
        records = [record for record in load_jsonl(path) if record.get("object_id") not in remove_ids]
        write_jsonl(path, records)

    split_report = dataset_root / "split_report.json"
    if split_report.exists():
        report = read_json(split_report)
        for split in report.get("splits", {}).values():
            ids = split.get("object_ids")
            if isinstance(ids, list):
                split["object_ids"] = [object_id for object_id in ids if object_id not in remove_ids]
                split["count"] = len(split["object_ids"])
            for key in ("categories", "tags"):
                if isinstance(split.get(key), dict):
                    for object_id in remove_ids:
                        split[key].pop(object_id, None)
        if isinstance(report.get("rejected"), list):
            report["rejected"] = [item for item in report["rejected"] if item.get("object_id") not in remove_ids]
        if isinstance(report.get("render_failures"), list):
            report["render_failures"] = [
                item for item in report["render_failures"] if item.get("object_id") not in remove_ids
            ]
        report["accepted_object_count"] = sum(
            split.get("count", 0) for split in report.get("splits", {}).values()
        )
        write_json(split_report, report)

    rename_map = dataset_root / "object_rename_map.json"
    if rename_map.exists():
        entries = read_json(rename_map)
        if isinstance(entries, list):
            entries = [entry for entry in entries if entry.get("new_object_id") not in remove_ids]
            write_json(rename_map, entries)


def load_manifest_record_map(dataset_root: Path) -> dict[str, dict]:
    records = {}
    for filename in ("train.jsonl", "val.jsonl", "test_objaverse_heldout.jsonl"):
        for record in load_jsonl(dataset_root / filename):
            object_id = record.get("object_id")
            if object_id and object_id not in records:
                records[object_id] = record
    return records


def candidate_from_meta_or_manifest(dataset_root: Path, object_id: str, manifest_records: dict[str, dict]):
    meta_path = object_dir(dataset_root, object_id) / "meta.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        source_path = Path(meta.get("source_path") or "")
        name = meta.get("name") or object_id
        tags = meta.get("tags") or []
        categories = meta.get("categories") or []
        original_object_id = meta.get("original_object_id") or object_id
    else:
        record = manifest_records.get(object_id)
        if not record:
            print(f"[repair:skip] {object_id}: missing meta.json and manifest record")
            return None
        metadata = record.get("metadata") or {}
        source_path = Path(metadata.get("source_path") or "")
        name = metadata.get("name") or object_id
        tags = metadata.get("tags") or []
        categories = metadata.get("categories") or []
        original_object_id = metadata.get("original_object_id") or object_id
        print(f"[repair:recover] {object_id}: using manifest metadata because meta.json is missing")

    if not source_path.exists():
        print(f"[repair:skip] {object_id}: missing source model {source_path}")
        return None

    return builder.Candidate(
        object_id=object_id,
        source_path=source_path,
        name=name,
        tags=tags,
        categories=categories,
        metadata={
            "object_id": original_object_id,
            "original_object_id": original_object_id,
        },
    )


def mesh_bbox(obj):
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))
    for corner in obj.bound_box:
        world = obj.matrix_world @ Vector(corner)
        min_corner.x = min(min_corner.x, world.x)
        min_corner.y = min(min_corner.y, world.y)
        min_corner.z = min(min_corner.z, world.z)
        max_corner.x = max(max_corner.x, world.x)
        max_corner.y = max(max_corner.y, world.y)
        max_corner.z = max(max_corner.z, world.z)
    return min_corner, max_corner


def bbox_volume(min_corner, max_corner) -> float:
    size = max_corner - min_corner
    return max(size.x, 1e-6) * max(size.y, 1e-6) * max(size.z, 1e-6)


def delete_object(obj) -> None:
    bpy.data.objects.remove(obj, do_unlink=True)


def remove_ground_plate_like_meshes() -> list[str]:
    objects = builder.mesh_objects()
    if len(objects) <= 1:
        return []
    global_min, global_max = builder.bbox_for_objects(objects)
    height = max(global_max.z - global_min.z, 1e-6)
    removed = []
    for obj in list(objects):
        min_corner, max_corner = mesh_bbox(obj)
        size = max_corner - min_corner
        thin = size.z <= height * 0.08
        near_bottom = max_corner.z <= global_min.z + height * 0.18
        wide = max(size.x, size.y) >= max(global_max.x - global_min.x, global_max.y - global_min.y) * 0.45
        if thin and near_bottom and wide:
            removed.append(obj.name)
            delete_object(obj)
    bpy.context.view_layer.update()
    return removed


def remove_far_secondary_meshes() -> list[str]:
    objects = builder.mesh_objects()
    if len(objects) <= 1:
        return []
    boxes = [(obj, *mesh_bbox(obj)) for obj in objects]
    main_obj, main_min, main_max = max(boxes, key=lambda item: bbox_volume(item[1], item[2]))
    main_center = (main_min + main_max) * 0.5
    main_size = main_max - main_min
    radius = max(main_size.x, main_size.y, main_size.z, 1e-6)
    removed = []
    for obj, min_corner, max_corner in boxes:
        if obj is main_obj:
            continue
        center = (min_corner + max_corner) * 0.5
        distance = math.sqrt((center.x - main_center.x) ** 2 + (center.y - main_center.y) ** 2)
        volume_ratio = bbox_volume(min_corner, max_corner) / max(bbox_volume(main_min, main_max), 1e-6)
        if distance > radius * 1.15 and volume_ratio < 0.35:
            removed.append(obj.name)
            delete_object(obj)
    bpy.context.view_layer.update()
    return removed


def apply_yaw_correction(yaw_degrees: float) -> None:
    if not yaw_degrees:
        return
    root = bpy.data.objects.new("RepairYawRoot", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in [obj for obj in bpy.context.scene.objects if obj.parent is None and obj is not root]:
        matrix = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = matrix
    root.rotation_euler[2] = math.radians(yaw_degrees)
    bpy.context.view_layer.update()


def render_repair(candidate: builder.Candidate, object_id: str, dataset_root: Path, args):
    object_path = object_dir(dataset_root, object_id)
    if object_path.exists():
        shutil.rmtree(object_path)
    object_path.mkdir(parents=True, exist_ok=True)

    builder.import_model(candidate.source_path)
    removed_meshes = []
    if object_id in REMOVE_GROUND_IDS:
        removed_meshes.extend(remove_ground_plate_like_meshes())
    if object_id in REMOVE_FAR_IDS:
        removed_meshes.extend(remove_far_secondary_meshes())
    apply_yaw_correction(YAW_DEGREES.get(object_id, 0.0))

    norm = builder.normalize_scene(args.target_size)
    camera_radius = builder.compute_camera_radius(norm, args.fov, args.radius, args.camera_padding)
    camera = builder.setup_scene(args.render_size, args.fov, white_background=args.render_white_background)
    views = [
        ("cond.jpg", 0, 0),
        *[(filename, azimuth, elevation) for _, filename, azimuth, elevation in builder.TARGET_POSES],
        *builder.REFERENCE_POSES,
    ]
    for filename, azimuth, elevation in views:
        raw_path = object_path / f".{Path(filename).stem}_raw.png"
        builder.point_camera(camera, azimuth, elevation, camera_radius, norm["look_at"])
        bpy.context.scene.render.filepath = str(raw_path)
        result = bpy.ops.render.render(write_still=True)
        if "FINISHED" not in result:
            raise RuntimeError(f"Blender render failed for {filename}: {result}")
        builder.postprocess_render(raw_path, object_path / filename, args)
        raw_path.unlink(missing_ok=True)

    builder.write_json(object_path / "ref_metadata.json", builder.ref_metadata())
    ok, reason, quality_reports = builder.validate_rendered_object(object_path, args)
    meta = {
        "object_id": object_id,
        "name": candidate.name,
        "tags": candidate.tags,
        "categories": candidate.categories,
        "source_path": str(candidate.source_path),
        "original_object_id": candidate.metadata.get("original_object_id") or candidate.metadata.get("object_id"),
        "target_order": builder.TARGET_ORDER,
        "ref_metadata": "ref_metadata.json",
        "validation_passed": ok,
        "validation_reason": reason,
        "quality_reports": quality_reports,
        "repair": {
            "yaw_degrees": YAW_DEGREES.get(object_id, 0.0),
            "removed_meshes": removed_meshes,
        },
    }
    builder.write_json(object_path / "meta.json", meta)
    if not ok:
        raise RuntimeError(reason)
    return builder.manifest_record(candidate, dataset_root)


def update_record(dataset_root: Path, object_id: str, replacement: dict) -> None:
    found = False
    for filename in ("train.jsonl", "val.jsonl", "test_objaverse_heldout.jsonl"):
        path = dataset_root / filename
        records = load_jsonl(path)
        changed = False
        for index, record in enumerate(records):
            if record.get("object_id") == object_id:
                records[index] = replacement
                found = True
                changed = True
        if changed:
            write_jsonl(path, records)
    if not found:
        records = load_jsonl(dataset_root / "train.jsonl")
        records.append(replacement)
        write_jsonl(dataset_root / "train.jsonl", records)


def main_blender(args) -> None:
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = REPO_ROOT / dataset_root
    dataset_root = dataset_root.resolve()
    object_ids = parse_object_ids(Path(args.quality_list))
    remove_ids = set(object_ids) & REMOVE_IDS
    rerender_ids = [object_id for object_id in object_ids if object_id not in remove_ids]

    print(f"[repair] remove IDs: {sorted(remove_ids)}")
    print(f"[repair] rerender IDs: {rerender_ids}")

    for object_id in sorted(remove_ids):
        path = object_dir(dataset_root, object_id)
        if path.exists():
            shutil.rmtree(path)
            print(f"[repair:removed] {object_id}: deleted {path}")
    remove_records(dataset_root, remove_ids)

    manifest_records = load_manifest_record_map(dataset_root)
    rejected = []
    for object_id in rerender_ids:
        candidate = candidate_from_meta_or_manifest(dataset_root, object_id, manifest_records)
        if candidate is None:
            continue
        print(
            f"[repair:render] {object_id}: source={candidate.source_path} "
            f"yaw={YAW_DEGREES.get(object_id, 0.0):g}"
        )
        try:
            record = render_repair(candidate, object_id, dataset_root, args)
            update_record(dataset_root, object_id, record)
        except Exception as exc:
            rejected.append({"object_id": object_id, "reason": str(exc)})
            print(f"[repair:rejected] {object_id}: {exc}")

    if rejected:
        rejected_path = dataset_root / "repair_rejected.json"
        write_json(rejected_path, rejected)
        print(f"[repair] rejected during repair: {rejected_path}")
    else:
        (dataset_root / "repair_rejected.json").unlink(missing_ok=True)
    print(f"[repair] complete: removed={len(remove_ids)} rerendered={len(rerender_ids) - len(rejected)} rejected={len(rejected)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data/rag_zero123plus_objaverse_toys_500")
    parser.add_argument("--quality_list", required=True)
    parser.add_argument("--blender_path", default=r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe")
    parser.add_argument("--postprocess_python", default=r"D:\conda_envs\instantmesh2\python.exe")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--render_size", type=int, default=1536)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--fov", type=float, default=30.0)
    parser.add_argument("--target_size", type=float, default=1.6)
    parser.add_argument("--camera_padding", type=float, default=1.18)
    parser.add_argument("--min_occupancy", type=float, default=0.03)
    parser.add_argument("--max_occupancy", type=float, default=0.92)
    parser.add_argument("--max_center_offset", type=float, default=0.18)
    parser.add_argument("--render_white_background", action="store_true")
    parser.add_argument("--adaptive_foreground_rescale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_fill_ratio", type=float, default=0.70)
    parser.add_argument("--target_fill_ratio", type=float, default=0.78)
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = sys.argv[1:]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if bpy is None:
        dataset_root = Path(args.dataset_root)
        if not dataset_root.is_absolute():
            dataset_root = REPO_ROOT / dataset_root
        quality_list = Path(args.quality_list)
        if not quality_list.is_absolute():
            quality_list = REPO_ROOT / quality_list
        cmd = [
            args.blender_path,
            "--background",
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "--dataset_root",
            str(dataset_root.resolve()),
            "--quality_list",
            str(quality_list.resolve()),
            "--postprocess_python",
            args.postprocess_python,
            "--image_size",
            str(args.image_size),
            "--render_size",
            str(args.render_size),
            "--radius",
            str(args.radius),
            "--fov",
            str(args.fov),
            "--target_size",
            str(args.target_size),
            "--camera_padding",
            str(args.camera_padding),
            "--min_occupancy",
            str(args.min_occupancy),
            "--max_occupancy",
            str(args.max_occupancy),
            "--max_center_offset",
            str(args.max_center_offset),
            "--min_fill_ratio",
            str(args.min_fill_ratio),
            "--target_fill_ratio",
            str(args.target_fill_ratio),
        ]
        if args.render_white_background:
            cmd.append("--render_white_background")
        if not args.adaptive_foreground_rescale:
            cmd.append("--no-adaptive_foreground_rescale")
        print("[repair-wrapper] launching Blender:")
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
        subprocess.run(cmd, check=True)
        return
    main_blender(args)


if __name__ == "__main__":
    main()
