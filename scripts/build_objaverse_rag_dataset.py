from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat
except ImportError:
    Image = None
    ImageChops = None
    ImageDraw = None
    ImageFont = None
    ImageStat = None

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None
    Vector = None


SUPPORTED_MODEL_EXTS = {".glb", ".gltf", ".obj", ".fbx"}
TARGET_POSES = [
    ("30", "target_030_el20.jpg", 30, 20),
    ("90", "target_090_el-10.jpg", 90, -10),
    ("150", "target_150_el20.jpg", 150, 20),
    ("210", "target_210_el-10.jpg", 210, -10),
    ("270", "target_270_el20.jpg", 270, 20),
    ("330", "target_330_el-10.jpg", 330, -10),
]
REFERENCE_POSES = [
    ("ref_090_el-10.jpg", 90, -10),
    ("ref_180_el0.jpg", 180, 0),
    ("ref_240_el-10.jpg", 240, -10),
]
TARGET_ORDER = ["30", "90", "150", "210", "270", "330"]


@dataclass
class Candidate:
    object_id: str
    source_path: Optional[Path] = None
    name: str = ""
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    description: str = ""
    metadata: Dict = field(default_factory=dict)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build an object-level Zero123++ RAG adapter dataset from Objaverse-style 3D assets."
    )
    parser.add_argument("--output_root", default="data/rag_zero123plus_objaverse_toys")
    parser.add_argument("--source_dir", default=None, help="Local folder containing .glb/.gltf/.obj/.fbx assets.")
    parser.add_argument("--metadata_path", default=None, help="Optional Objaverse-style metadata JSON/JSONL.")
    parser.add_argument("--download_objaverse", action="store_true", help="Download Objaverse assets matching the keyword filter before rendering.")
    parser.add_argument("--download_only", action="store_true", help="Only download Objaverse assets and write metadata; do not build manifests.")
    parser.add_argument("--objaverse_cache_dir", default=r"D:\objaverse_cache", help="Objaverse package cache directory.")
    parser.add_argument("--objaverse_download_processes", type=int, default=4)
    parser.add_argument("--objaverse_candidate_pool", type=int, default=3000, help="Maximum matching Objaverse annotations to consider before sampling/downloading.")
    parser.add_argument(
        "--seed_dataset_root",
        default=None,
        help="Optional existing rendered RAG dataset to copy into the output before rendering new assets.",
    )
    parser.add_argument("--category_keywords", nargs="*", default=[])
    parser.add_argument("--max_objects", type=int, default=1000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--render_size", type=int, default=768)
    parser.add_argument("--render_white_background", action="store_true")
    parser.add_argument("--blender_path", default="blender")
    parser.add_argument("--postprocess_python", default=None)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--fov", type=float, default=30.0)
    parser.add_argument("--target_size", type=float, default=1.6)
    parser.add_argument("--camera_padding", type=float, default=1.18, help="Larger values zoom out to reduce cropping.")
    parser.add_argument("--adaptive_foreground_rescale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_fill_ratio", type=float, default=0.70, help="Only foregrounds below this max-side fill ratio are enlarged.")
    parser.add_argument("--target_fill_ratio", type=float, default=0.78, help="Adaptive enlargement target max-side fill ratio.")
    parser.add_argument("--min_occupancy", type=float, default=0.03)
    parser.add_argument("--max_occupancy", type=float, default=0.92)
    parser.add_argument("--max_center_offset", type=float, default=0.18)
    parser.add_argument("--preview_count", type=int, default=24)
    parser.add_argument("--skip_render", action="store_true", help="Only inspect existing rendered object folders.")
    parser.add_argument("--allow_empty", action="store_true", help="Write empty reports instead of failing when no assets are found.")
    parser.add_argument("--clip_filter_text", default=None, help="Optional text prompt for CLIP filtering when dependencies are available.")
    parser.add_argument("--clip_filter_min_score", type=float, default=None)
    argv = _blender_argv(argv)
    args = parser.parse_args(argv)
    if args.postprocess_python is None:
        preferred = Path(r"D:\conda_envs\instantmesh2\python.exe")
        args.postprocess_python = str(preferred) if preferred.exists() else sys.executable
    return args


def _blender_argv(argv):
    if argv is not None:
        return argv
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def slug_object_id(value: str, fallback_index: int) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:80] or f"object_{fallback_index:06d}"


def readable_objaverse_id(uid: str, annotation: Dict, index: int) -> str:
    name = annotation.get("name") or annotation.get("title") or annotation.get("description") or "objaverse_object"
    name_slug = slug_object_id(str(name), index)[:64]
    uid_slug = slug_object_id(str(uid), index)[:12]
    if uid_slug and uid_slug not in name_slug:
        return f"{name_slug}_{uid_slug}"
    return name_slug


def load_metadata_records(path: Optional[str]) -> List[Dict]:
    if not path:
        return []
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if metadata_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "objects" in raw and isinstance(raw["objects"], list):
            return raw["objects"]
        return [dict({"object_id": key}, **value) if isinstance(value, dict) else {"object_id": key, "value": value}
                for key, value in raw.items()]
    raise ValueError("Objaverse metadata must be a JSON object, list, or JSONL file.")


def discover_local_models(source_dir: Optional[str]) -> List[Path]:
    if not source_dir:
        return []
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")
    return [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_EXTS]


def candidate_from_metadata(record: Dict, index: int, source_lookup: Dict[str, Path]) -> Candidate:
    object_id = record.get("object_id") or record.get("uid") or record.get("id") or record.get("sha256") or index
    local_path = record.get("local_path") or record.get("path") or record.get("file") or record.get("model_path")
    source_path = Path(local_path) if local_path else None
    if source_path is not None and not source_path.exists():
        source_path = source_lookup.get(Path(local_path).name) or source_lookup.get(Path(local_path).stem)
    name = str(record.get("name") or record.get("title") or (source_path.stem if source_path else object_id))
    tags = _string_list(record.get("tags") or record.get("tag") or [])
    categories = _string_list(record.get("categories") or record.get("category") or record.get("labels") or [])
    description = str(record.get("description") or record.get("caption") or "")
    return Candidate(
        object_id=slug_object_id(object_id, index),
        source_path=source_path,
        name=name,
        tags=tags,
        categories=categories,
        description=description,
        metadata=record,
    )


def _string_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(key) for key, enabled in value.items() if enabled]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def build_candidates(source_dir: Optional[str], metadata_path: Optional[str]) -> List[Candidate]:
    source_paths = discover_local_models(source_dir)
    source_lookup = {path.name: path for path in source_paths}
    source_lookup.update({path.stem: path for path in source_paths})
    records = load_metadata_records(metadata_path)
    if records:
        candidates = [candidate_from_metadata(record, idx, source_lookup) for idx, record in enumerate(records, start=1)]
        return [candidate for candidate in candidates if candidate.source_path is None or candidate.source_path.exists()]
    return [
        Candidate(object_id=slug_object_id(path.stem, idx), source_path=path, name=path.stem)
        for idx, path in enumerate(source_paths, start=1)
    ]


def _annotation_values(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values = []
        for key, item in value.items():
            values.append(str(key))
            values.extend(_annotation_values(item))
        return values
    if isinstance(value, Iterable):
        values = []
        for item in value:
            values.extend(_annotation_values(item))
        return values
    return [str(value)]


def objaverse_annotation_text(uid: str, annotation: Dict) -> str:
    fields = [uid]
    for key in ("name", "title", "description", "caption", "tags", "categories", "labels"):
        fields.extend(_annotation_values(annotation.get(key)))
    return " ".join(fields).lower()


def download_objaverse_assets(args) -> Path:
    try:
        import objaverse
    except ImportError as exc:
        raise RuntimeError(
            "The Python environment does not have the `objaverse` package. Install it in the conda env "
            "before downloading, for example: pip install objaverse"
        ) from exc

    if not args.source_dir:
        raise RuntimeError("--download_objaverse requires --source_dir so downloaded asset metadata has a stable D: location.")
    os.environ.setdefault("OBJAVERSE_HOME", str(args.objaverse_cache_dir))
    source_dir = Path(args.source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    print(f"[objaverse-rag] loading Objaverse annotations into cache {os.environ['OBJAVERSE_HOME']}")
    annotations = objaverse.load_annotations()
    keywords = [keyword.lower().replace("_", " ") for keyword in args.category_keywords]
    matching = []
    for uid, annotation in annotations.items():
        text = objaverse_annotation_text(uid, annotation).replace("_", " ")
        if not keywords or any(keyword in text for keyword in keywords):
            matching.append((uid, annotation))
        if len(matching) >= int(args.objaverse_candidate_pool):
            break
    random.Random(args.seed).shuffle(matching)
    download_count = min(max(0, int(args.max_objects)), len(matching))
    selected = matching[:download_count]
    uids = [uid for uid, _ in selected]
    if not uids:
        raise RuntimeError("No Objaverse annotations matched the requested keywords.")
    print(f"[objaverse-rag] downloading {len(uids)} Objaverse assets")
    paths = objaverse.load_objects(uids=uids, download_processes=int(args.objaverse_download_processes))
    records = []
    used_names = set()
    for index, (uid, annotation) in enumerate(selected, start=1):
        local_path = paths.get(uid)
        if not local_path:
            continue
        local_path = Path(local_path)
        object_id = readable_objaverse_id(uid, annotation, index)
        base_object_id = object_id
        dedupe = 2
        while object_id in used_names:
            object_id = f"{base_object_id}_{dedupe}"
            dedupe += 1
        used_names.add(object_id)
        stable_path = source_dir / f"{object_id}{local_path.suffix.lower() or '.glb'}"
        if not stable_path.exists():
            shutil.copy2(local_path, stable_path)
        records.append({
            "object_id": object_id,
            "uid": uid,
            "local_path": str(stable_path),
            "name": annotation.get("name") or annotation.get("title") or uid,
            "description": annotation.get("description") or annotation.get("caption") or "",
            "tags": _annotation_values(annotation.get("tags")),
            "categories": _annotation_values(annotation.get("categories") or annotation.get("labels")),
        })
    metadata_path = source_dir / "objaverse_downloaded_metadata.json"
    write_json(metadata_path, records)
    print(f"[objaverse-rag] wrote downloaded metadata: {metadata_path}")
    return metadata_path


def candidate_text(candidate: Candidate) -> str:
    return " ".join([candidate.object_id, candidate.name, candidate.description, *candidate.tags, *candidate.categories]).lower()


def filter_candidates(candidates: Sequence[Candidate], keywords: Sequence[str]) -> Tuple[List[Candidate], List[Dict]]:
    if not keywords:
        return list(candidates), []
    lowered = [keyword.lower() for keyword in keywords]
    accepted = []
    rejected = []
    for candidate in candidates:
        text = candidate_text(candidate)
        if any(keyword in text for keyword in lowered):
            accepted.append(candidate)
        else:
            rejected.append({"object_id": candidate.object_id, "reason": "keyword_filter", "name": candidate.name})
    return accepted, rejected


def maybe_clip_filter(candidates: Sequence[Candidate], args) -> Tuple[List[Candidate], List[Dict]]:
    if not args.clip_filter_text or args.clip_filter_min_score is None:
        return list(candidates), []
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return list(candidates), [{
            "object_id": "__clip_filter__",
            "reason": "clip_filter_unavailable",
            "message": "Install torch/transformers and provide rendered preview images before enabling CLIP filtering.",
        }]
    return list(candidates), [{
        "object_id": "__clip_filter__",
        "reason": "clip_filter_not_applied",
        "message": "CLIP hooks are available, but this builder only filters metadata before rendering by default.",
    }]


def object_root(output_root: Path, object_id: str) -> Path:
    return output_root / "objects" / object_id


def ref_metadata() -> Dict[str, Dict[str, float]]:
    return {filename: {"azimuth": azimuth, "elevation": elevation} for filename, azimuth, elevation in REFERENCE_POSES}


def manifest_record(candidate: Candidate, root: Path) -> Dict:
    object_id = candidate.object_id
    return {
        "object_id": object_id,
        "cond_img": f"objects/{object_id}/cond.jpg",
        "target_azimuths": [azimuth for _, _, azimuth, _ in TARGET_POSES],
        "target_elevations": [elevation for _, _, _, elevation in TARGET_POSES],
        "target_imgs": {
            key: f"objects/{object_id}/{filename}"
            for key, filename, _, _ in TARGET_POSES
        },
        "ref_imgs": [f"objects/{object_id}/{filename}" for filename, _, _ in REFERENCE_POSES],
        "ref_view_labels": ["unknown"] * len(REFERENCE_POSES),
        "ref_azimuths": [azimuth for _, azimuth, _ in REFERENCE_POSES],
        "ref_elevations": [elevation for _, _, elevation in REFERENCE_POSES],
        "metadata": {
            "name": candidate.name,
            "tags": candidate.tags,
            "categories": candidate.categories,
            "source_path": str(candidate.source_path) if candidate.source_path else None,
        },
    }


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copied_seed_record(record: Dict, seed_object_id: str) -> Dict:
    copied = dict(record)
    copied["object_id"] = seed_object_id
    copied["cond_img"] = f"objects/{seed_object_id}/{Path(record['cond_img']).name}"
    copied["target_imgs"] = {
        key: f"objects/{seed_object_id}/{Path(value).name}"
        for key, value in record["target_imgs"].items()
    }
    copied["ref_imgs"] = [
        f"objects/{seed_object_id}/{Path(value).name}"
        for value in record.get("ref_imgs", [])
    ]
    metadata = dict(copied.get("metadata", {}))
    metadata["seed_source"] = True
    copied["metadata"] = metadata
    return copied


def copy_seed_dataset(seed_root: Optional[str], output_root: Path) -> Tuple[List[Dict], set]:
    if not seed_root:
        return [], set()
    seed_root = Path(seed_root)
    records = load_jsonl(seed_root / "train.jsonl")
    copied_records = []
    copied_source_models = set()
    for record in records:
        old_object_id = record["object_id"]
        new_object_id = f"seed_{old_object_id}"
        src_dir = seed_root / old_object_id
        dst_dir = object_root(output_root, new_object_id)
        if not src_dir.exists():
            continue
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        meta_path = dst_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                meta = {}
        else:
            meta = {}
        source_model = meta.get("source_model")
        if source_model:
            copied_source_models.add(str(Path(source_model).resolve()).lower())
        meta.update({
            "object_id": new_object_id,
            "copied_from_seed_dataset": str(seed_root),
            "seed_object_id": old_object_id,
        })
        write_json(meta_path, meta)
        copied_records.append(copied_seed_record(record, new_object_id))
    return copied_records, copied_source_models


def foreground_bbox(image: Image.Image, threshold: int = 8):
    if Image is None:
        raise RuntimeError("Pillow is required for direct image analysis.")
    rgb = image.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
    channels = diff.split()
    mask = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    return mask.point(lambda value: 255 if value > threshold else 0).getbbox()


def image_quality_report(path: Path) -> Dict:
    report_path = path.with_name(f".{path.stem}_quality.json")
    if Image is None and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["path"] = str(path)
        report.setdefault("mode", "RGB")
        report.setdefault("has_alpha", False)
        report.setdefault("center_offset", 0.0)
        report.setdefault("corner_white", True)
        return report
    if Image is None:
        raise RuntimeError(
            "Pillow is not installed in this Python. Run through Blender with "
            "--postprocess_python pointing to the conda env so quality JSON files are created."
        )
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
    width, height = rgb.size
    bbox = foreground_bbox(rgb)
    if bbox is None:
        bbox_width = bbox_height = 0
        occupancy = 0.0
        center_offset = 1.0
        luminance = 0.0
    else:
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        occupancy = (bbox_width * bbox_height) / float(width * height)
        bbox_center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        center_offset = math.sqrt(
            ((bbox_center[0] / width) - 0.5) ** 2 + ((bbox_center[1] / height) - 0.5) ** 2
        )
        luminance = ImageStat.Stat(rgb.convert("L").crop(bbox)).mean[0] / 255.0
    corner_samples = [
        rgb.crop((0, 0, 12, 12)),
        rgb.crop((width - 12, 0, width, 12)),
        rgb.crop((0, height - 12, 12, height)),
        rgb.crop((width - 12, height - 12, width, height)),
    ]
    corner_mean = sum(sum(ImageStat.Stat(crop).mean) / 3.0 for crop in corner_samples) / len(corner_samples)
    corner_white = corner_mean >= 245.0
    return {
        "path": str(path),
        "resolution": [width, height],
        "mode": rgb.mode,
        "has_alpha": False,
        "bbox": list(bbox) if bbox else None,
        "bbox_width_ratio": bbox_width / float(width),
        "bbox_height_ratio": bbox_height / float(height),
        "bbox_occupancy": occupancy,
        "center_offset": center_offset,
        "foreground_luminance": luminance,
        "corner_white": corner_white,
    }


def validate_quality(report: Dict, args) -> Tuple[bool, str]:
    if report["mode"] != "RGB":
        return False, f"expected RGB image, got {report['mode']}"
    if report["resolution"] != [args.image_size, args.image_size]:
        return False, f"resolution mismatch {report['resolution']}"
    if report["bbox"] is None or report["bbox_occupancy"] <= 0:
        return False, "mostly empty or no foreground found"
    if report["bbox_occupancy"] < args.min_occupancy:
        return False, f"object too small occupancy={report['bbox_occupancy']:.4f}"
    if report["bbox_occupancy"] > args.max_occupancy:
        return False, f"object too large/cropped occupancy={report['bbox_occupancy']:.4f}"
    if report["center_offset"] > args.max_center_offset:
        return False, f"object not centered center_offset={report['center_offset']:.4f}"
    if not report["corner_white"]:
        return False, "background corners are not white"
    return True, "ok"


def validate_rendered_object(object_dir: Path, args) -> Tuple[bool, str, Dict]:
    required = ["cond.jpg", *[filename for _, filename, _, _ in TARGET_POSES], *[filename for filename, _, _ in REFERENCE_POSES]]
    reports = {}
    blank_or_broken = 0
    for filename in required:
        path = object_dir / filename
        if not path.exists():
            return False, f"missing {filename}", reports
        try:
            report = image_quality_report(path)
        except Exception as exc:
            return False, f"broken image {filename}: {exc}", reports
        reports[filename] = report
        ok, _ = validate_quality(report, args)
        if not ok:
            blank_or_broken += 1
    if blank_or_broken > 0:
        first_bad = next(
            (filename for filename, report in reports.items() if not validate_quality(report, args)[0]),
            required[0],
        )
        return False, f"{first_bad}: {validate_quality(reports[first_bad], args)[1]}", reports
    return True, "ok", reports


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def split_records(records: Sequence[Dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_count = int(round(count * train_ratio / total_ratio))
    val_count = int(round(count * val_ratio / total_ratio))
    if train_count + val_count > count:
        val_count = max(0, count - train_count)
    train = shuffled[:train_count]
    val = shuffled[train_count:train_count + val_count]
    test = shuffled[train_count + val_count:]
    return {"train": train, "val": val, "test_objaverse_heldout": test}


def assert_no_split_overlap(splits: Dict[str, Sequence[Dict]]) -> None:
    seen = {}
    for split_name, records in splits.items():
        for record in records:
            object_id = record["object_id"]
            if object_id in seen:
                raise ValueError(f"Object ID {object_id} appears in both {seen[object_id]} and {split_name}")
            seen[object_id] = split_name


def write_split_manifests(output_root: Path, splits: Dict[str, Sequence[Dict]]) -> None:
    write_jsonl(output_root / "train.jsonl", splits["train"])
    write_jsonl(output_root / "val.jsonl", splits["val"])
    write_jsonl(output_root / "test_objaverse_heldout.jsonl", splits["test_objaverse_heldout"])


def copy_rejected_example(object_dir: Path, rejected_dir: Path, object_id: str, reason: str, metadata: Dict) -> None:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    write_json(rejected_dir / f"{object_id}.json", {"object_id": object_id, "reason": reason, **metadata})
    if object_dir.exists():
        for image_path in object_dir.glob("*.jpg"):
            target = rejected_dir / f"{object_id}_{image_path.name}"
            try:
                shutil.copy2(image_path, target)
            except OSError:
                pass


def create_preview_grid(output_root: Path, records: Sequence[Dict], max_items: int = 24) -> None:
    if Image is None:
        return
    selected = list(records)[:max_items]
    if not selected:
        return
    thumbs = []
    labels = []
    for record in selected:
        path = output_root / record["cond_img"]
        if not path.exists():
            continue
        with Image.open(path) as image:
            thumbs.append(image.convert("RGB").resize((120, 120), Image.Resampling.LANCZOS))
        labels.append(record["object_id"][:18])
    if not thumbs:
        return
    cols = min(6, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    font = ImageFont.load_default()
    grid = Image.new("RGB", (cols * 120, rows * 144), "white")
    draw = ImageDraw.Draw(grid)
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 120
        y = (idx // cols) * 144
        grid.paste(thumb, (x, y))
        draw.text((x + 4, y + 123), labels[idx], fill="black", font=font)
    grid.save(output_root / "dataset_preview_grid.png")


def write_external_test_readme(output_root: Path) -> None:
    external = output_root / "external_test"
    external.mkdir(parents=True, exist_ok=True)
    (external / "README.md").write_text(
        "Place manually collected plushie/toy examples here for final external evaluation. "
        "Keep object IDs separate from Objaverse train/val/test objects.\n",
        encoding="utf-8",
    )


def render_candidate(candidate: Candidate, object_dir: Path, args) -> None:
    if args.skip_render:
        return
    if candidate.source_path is None:
        raise RuntimeError("No local source_path is available for rendering.")
    if bpy is None:
        raise RuntimeError(
            "Rendering requires Blender. Re-run with Blender, for example: "
            f"{args.blender_path} --background --python scripts/build_objaverse_rag_dataset.py -- --source_dir <models> "
            f"--output_root {args.output_root}"
        )
    _render_candidate_with_blender(candidate, object_dir, args)


def _render_candidate_with_blender(candidate: Candidate, object_dir: Path, args) -> None:
    import_model(candidate.source_path)
    norm = normalize_scene(args.target_size)
    camera_radius = compute_camera_radius(norm, args.fov, args.radius, args.camera_padding)
    camera = setup_scene(args.render_size, args.fov, white_background=args.render_white_background)
    views = [("cond.jpg", 0, 0), *[(filename, az, el) for _, filename, az, el in TARGET_POSES], *REFERENCE_POSES]
    for filename, azimuth, elevation in views:
        raw_path = object_dir / f".{Path(filename).stem}_raw.png"
        point_camera(camera, azimuth, elevation, camera_radius, norm["look_at"])
        bpy.context.scene.render.filepath = str(raw_path)
        result = bpy.ops.render.render(write_still=True)
        if "FINISHED" not in result:
            raise RuntimeError(f"Blender render failed for {filename}: {result}")
        postprocess_render(raw_path, object_dir / filename, args)
        raw_path.unlink(missing_ok=True)


def import_model(path: Path) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    ext = path.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        raise ValueError(f"Unsupported model extension: {path}")


def mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def bbox_for_objects(objects):
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))
    bpy.context.view_layer.update()
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world.x)
            min_corner.y = min(min_corner.y, world.y)
            min_corner.z = min(min_corner.z, world.z)
            max_corner.x = max(max_corner.x, world.x)
            max_corner.y = max(max_corner.y, world.y)
            max_corner.z = max(max_corner.z, world.z)
    return min_corner, max_corner


def normalize_scene(target_size: float) -> Dict:
    objects = mesh_objects()
    if not objects:
        raise RuntimeError("Imported asset contains no mesh objects.")
    min_corner, max_corner = bbox_for_objects(objects)
    center = (min_corner + max_corner) * 0.5
    size = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z, 1e-6)
    scale = target_size / size
    root = bpy.data.objects.new("RAGDatasetRoot", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in [obj for obj in bpy.context.scene.objects if obj.parent is None and obj is not root]:
        matrix = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = matrix
    root.scale = (scale, scale, scale)
    root.location = (-center.x * scale, -center.y * scale, -min_corner.z * scale)
    bpy.context.view_layer.update()
    min_after, max_after = bbox_for_objects(objects)
    center_after = (min_after + max_after) * 0.5
    size_after = max_after - min_after
    return {
        "look_at": list(center_after),
        "bbox_size_after": list(size_after),
    }


def compute_camera_radius(norm: Dict, fov_degrees: float, min_radius: float, padding: float) -> float:
    size = norm["bbox_size_after"]
    diagonal = math.sqrt(size[0] ** 2 + size[1] ** 2 + size[2] ** 2)
    bounding_radius = max(diagonal * 0.5, 1e-6)
    half_fov = math.radians(max(fov_degrees, 1e-3) * 0.5)
    fit_radius = bounding_radius / max(math.sin(half_fov), 1e-6)
    return max(float(min_radius), fit_radius * float(padding))


def setup_scene(render_size: int, fov: float, white_background: bool):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.render.resolution_x = render_size
    scene.render.resolution_y = render_size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (1, 1, 1)
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera.data.angle = math.radians(fov)
    for name, loc, energy in [
        ("Front_Key", (0, -4.5, 4.5), 480),
        ("Rear_Fill", (0, 4.5, 3.2), 300),
        ("Left_Fill", (-4, 0, 3), 250),
        ("Right_Fill", (4, 0, 3), 250),
    ]:
        light_data = bpy.data.lights.new(name, type="AREA")
        light = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light)
        light.location = loc
        light.data.energy = energy
        light.data.size = 4.0
    return camera


def point_camera(camera, azimuth: float, elevation: float, radius: float, look_at) -> None:
    az = math.radians(azimuth)
    el = math.radians(elevation)
    look_at = Vector(look_at)
    camera.location = look_at + Vector((
        radius * math.cos(el) * math.sin(az),
        -radius * math.cos(el) * math.cos(az),
        radius * math.sin(el),
    ))
    direction = look_at - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def postprocess_render(raw_path: Path, output_path: Path, args) -> None:
    helper = Path(__file__).resolve().parent / "postprocess_rag_render.py"
    report_path = output_path.with_name(f".{output_path.stem}_quality.json")
    subprocess.run([
        args.postprocess_python,
        str(helper),
        "--input", str(raw_path),
        "--output", str(output_path),
        "--size", str(args.image_size),
        "--report", str(report_path),
        *(
            [
                "--adaptive_rescale",
                "--min_fill_ratio", str(args.min_fill_ratio),
                "--target_fill_ratio", str(args.target_fill_ratio),
            ]
            if args.adaptive_foreground_rescale else []
        ),
    ], check=True)


def build_dataset(args) -> Dict:
    output_root = Path(args.output_root)
    objects_dir = output_root / "objects"
    rejected_dir = output_root / "rejected"
    objects_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    write_external_test_readme(output_root)

    seed_records, seed_source_models = copy_seed_dataset(args.seed_dataset_root, output_root)
    candidates = build_candidates(args.source_dir, args.metadata_path)
    if seed_source_models:
        candidates = [
            candidate for candidate in candidates
            if candidate.source_path is None
            or str(candidate.source_path.resolve()).lower() not in seed_source_models
        ]
    keyword_candidates, keyword_rejected = filter_candidates(candidates, args.category_keywords)
    filtered_candidates, clip_rejected = maybe_clip_filter(keyword_candidates, args)
    remaining_slots = max(0, int(args.max_objects) - len(seed_records))
    filtered_candidates = filtered_candidates[:remaining_slots]
    accepted_records = list(seed_records)
    rejected = [*keyword_rejected, *clip_rejected]
    render_failures = []

    if not filtered_candidates and not seed_records and not args.allow_empty:
        raise RuntimeError(
            "No Objaverse-style assets were found after filtering. Provide --source_dir with local "
            ".glb/.gltf/.obj/.fbx files or --metadata_path with local_path entries. The script does not "
            "download large Objaverse assets automatically."
        )
    if filtered_candidates and bpy is None and not args.skip_render:
        raise RuntimeError(
            "Rendering local 3D assets requires Blender's Python module (bpy). You ran this with normal "
            "Python, so no images were rendered. Re-run through Blender, or use --skip_render only when "
            "objects/<object_id>/ already contains cond.jpg, target_*.jpg, and ref_*.jpg."
        )

    for candidate in filtered_candidates:
        obj_dir = object_root(output_root, candidate.object_id)
        obj_dir.mkdir(parents=True, exist_ok=True)
        try:
            render_candidate(candidate, obj_dir, args)
            write_json(obj_dir / "ref_metadata.json", ref_metadata())
            meta = {
                "object_id": candidate.object_id,
                "name": candidate.name,
                "tags": candidate.tags,
                "categories": candidate.categories,
                "source_path": str(candidate.source_path) if candidate.source_path else None,
                "target_order": TARGET_ORDER,
                "ref_metadata": "ref_metadata.json",
            }
            ok, reason, quality_reports = validate_rendered_object(obj_dir, args)
            meta["validation_passed"] = ok
            meta["validation_reason"] = reason
            meta["quality_reports"] = quality_reports
            write_json(obj_dir / "meta.json", meta)
            if not ok:
                rejected.append({"object_id": candidate.object_id, "reason": reason, "name": candidate.name})
                copy_rejected_example(obj_dir, rejected_dir, candidate.object_id, reason, meta)
                continue
            accepted_records.append(manifest_record(candidate, output_root))
        except Exception as exc:
            failure = {
                "object_id": candidate.object_id,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
                "name": candidate.name,
            }
            render_failures.append(failure)
            rejected.append(failure)
            copy_rejected_example(obj_dir, rejected_dir, candidate.object_id, str(exc), failure)

    splits = split_records(accepted_records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    assert_no_split_overlap(splits)
    write_split_manifests(output_root, splits)
    create_preview_grid(output_root, accepted_records, max_items=args.preview_count)

    split_report = {
        "output_root": str(output_root),
        "max_objects_requested": int(args.max_objects),
        "seed_dataset_root": args.seed_dataset_root,
        "seed_object_count": len(seed_records),
        "new_render_candidate_count": len(filtered_candidates),
        "accepted_object_count": len(accepted_records),
        "rejected_object_count": len(rejected),
        "render_failure_count": len(render_failures),
        "splits": {
            split_name: {
                "count": len(records),
                "object_ids": [record["object_id"] for record in records],
                "categories": {
                    record["object_id"]: record.get("metadata", {}).get("categories", [])
                    for record in records
                },
                "tags": {
                    record["object_id"]: record.get("metadata", {}).get("tags", [])
                    for record in records
                },
            }
            for split_name, records in splits.items()
        },
        "rejected": rejected,
        "render_failures": render_failures,
        "target_view_count": len(TARGET_POSES),
        "references_per_object": len(REFERENCE_POSES),
        "pretraining_overlap_note": (
            "Objaverse held-out evaluation is internal only; Zero123++ may have seen similar "
            "Objaverse-style data during pretraining."
        ),
    }
    write_json(output_root / "split_report.json", split_report)
    return split_report


def print_summary(report: Dict) -> None:
    splits = report["splits"]
    output_root = str(report["output_root"])
    if report.get("max_objects_requested") == 500 or output_root.endswith("rag_zero123plus_objaverse_toys_500"):
        train_config = "configs/zero123plus-rag-adapter-objaverse-wide-500obj-500steps-val.yaml"
    else:
        train_config = "configs/zero123plus-rag-adapter-objaverse-wide-1000.yaml"
    print("[objaverse-rag] dataset build summary")
    print(f"- output: {output_root}")
    print(f"- accepted objects: {report['accepted_object_count']}")
    print(f"- copied seed objects: {report.get('seed_object_count', 0)}")
    print(f"- new render candidates: {report.get('new_render_candidate_count', 0)}")
    print(f"- rejected objects: {report['rejected_object_count']}")
    print(f"- render failures: {report['render_failure_count']}")
    print(f"- train/val/test: {splits['train']['count']}/{splits['val']['count']}/{splits['test_objaverse_heldout']['count']}")
    print(f"- references per object: {report['references_per_object']}")
    print(f"- target views per object: {report['target_view_count']}")
    print("[objaverse-rag] suggested training command:")
    print(f"python train.py --base {train_config} --gpus 0 --num_nodes 1")


def main(argv=None):
    args = parse_args(argv)
    if args.download_objaverse:
        try:
            metadata_path = download_objaverse_assets(args)
        except RuntimeError as exc:
            print(f"[objaverse-rag:error] {exc}")
            return 2
        args.metadata_path = str(metadata_path)
        if args.download_only:
            seed_dataset_root = args.seed_dataset_root or r"D:\InstantMesh2\data\rag_zero123plus_tiny"
            print("[objaverse-rag] download_only=true; run Blender next with:")
            print(
                "& \"C:\\Program Files\\Blender Foundation\\Blender 4.4\\blender.exe\" "
                "--background --python scripts\\build_objaverse_rag_dataset.py -- "
                f"--output_root {args.output_root} "
                f"--seed_dataset_root {seed_dataset_root} "
                f"--source_dir {args.source_dir} "
                f"--metadata_path {metadata_path} "
                "--max_objects 500 --train_ratio 0.9 --val_ratio 0.1 --render_white_background --image_size 320 "
                "--postprocess_python D:\\conda_envs\\instantmesh2\\python.exe"
            )
            return 0
    try:
        report = build_dataset(args)
    except RuntimeError as exc:
        print(f"[objaverse-rag:error] {exc}")
        print("[objaverse-rag:setup] Install/use Blender for rendering local 3D assets, and provide --source_dir or --metadata_path.")
        if args.allow_empty:
            return 0
        return 2
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
