from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SLOT_AZIMUTHS = [30, 90, 150, 210, 270, 330]
ANGLE_TO_SLOT = {angle: slot for slot, angle in enumerate(SLOT_AZIMUTHS)}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
GT_TERMS = (
    "ground_truth",
    "ground-truth",
    "ground truth",
    "gt",
    "target",
    "targets",
    "reference_render",
    "reference-renders",
    "reference renders",
    "rendered_views",
    "rendered-views",
    "rendered views",
)


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int
    height: int
    mode: str
    is_sheet: bool
    tile_width: int | None
    tile_height: int | None
    sheet_error: str | None


@dataclass(frozen=True)
class GeneratedImage:
    public_object_dir: Path
    public_object_key: str
    object_id: str | None
    category: str | None
    blind_result: str
    method: str
    raw_method: str
    path: Path
    kind: str


@dataclass(frozen=True)
class GroundTruthImage:
    object_id: str
    slot: int | None
    azimuth: int | None
    path: Path
    source: str
    kind: str


def rel(path: Path, base: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def normalized_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def slug(text: str) -> str:
    text = text.strip().lower().replace("+", "plus")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def find_user_study_dir(start: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"User-study directory does not exist: {path}")
        return path

    candidates: list[Path] = []
    for path in [start.resolve(), *start.resolve().parents]:
        if "userstudy" in normalized_name(path.name):
            candidates.append(path)
    if candidates:
        return candidates[0]

    search_roots = [start.resolve()]
    if REPO_ROOT.exists():
        search_roots.append(REPO_ROOT)
        outputs = REPO_ROOT / "outputs"
        if outputs.exists():
            search_roots.insert(0, outputs)

    seen: set[Path] = set()
    matches: list[Path] = []
    for root in search_roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for path in root.rglob("*"):
            if path.is_dir() and "userstudy" in normalized_name(path.name):
                matches.append(path)
    if not matches:
        raise FileNotFoundError("Could not find a folder named like user_study, user-study, or user study.")
    matches.sort(key=lambda p: (len(p.parts), str(p)))
    return matches[0].resolve()


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_answer_key(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    mapping: dict[str, dict[str, str]] = defaultdict(dict)
    current_object: str | None = None
    heading_re = re.compile(r"^##\s+(.+?)\s*$")
    result_re = re.compile(r"^-\s*Result\s*([0-9]+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        heading = heading_re.match(line)
        if heading:
            current_object = heading.group(1).strip()
            continue
        result = result_re.match(line)
        if current_object and result:
            mapping[current_object][f"result_{int(result.group(1))}"] = result.group(2).strip()
    return dict(mapping)


def load_public_object_overrides(study_dir: Path) -> dict[str, str]:
    candidates = [
        study_dir / "public_object_overrides.csv",
        study_dir / "lpips_results" / "public_object_overrides.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        overrides: dict[str, str] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and {"public_object_key", "object_id"}.issubset(reader.fieldnames):
                for row in reader:
                    public_key = (row.get("public_object_key") or "").strip()
                    object_id = (row.get("object_id") or "").strip()
                    if public_key and object_id:
                        overrides[public_key] = object_id
            else:
                handle.seek(0)
                plain_reader = csv.reader(handle)
                for row in plain_reader:
                    if len(row) >= 2 and row[0].strip() and row[1].strip():
                        overrides[row[0].strip()] = row[1].strip()
        if overrides:
            return overrides
    return {}


def canonical_method(raw: str) -> str:
    compact = raw.lower()
    compact = compact.replace("zero123++", "zero123plus").replace("zero123+", "zero123plus")
    if "base" in compact or "baseline" in compact or "original" in compact or "zero123plus" in compact:
        return "base_zero123plus"
    if "epoch 3" in compact or "epoch_3" in compact or "step_0874" in compact:
        return "adapter_epoch_3"
    if "epoch 5" in compact or "epoch_5" in compact or "step_2500" in compact:
        return "adapter_epoch_5"
    if "last" in compact or "adapter_last" in compact:
        return "adapter_last"
    step_match = re.search(r"step[_-]?([0-9]+)", compact)
    if step_match:
        return f"step_{int(step_match.group(1)):04d}"
    return slug(raw)


def inspect_image(path: Path) -> ImageInfo:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    is_sheet = False
    tile_width = None
    tile_height = None
    sheet_error = None
    if width % 2 != 0:
        sheet_error = f"width {width} is not divisible by 2"
    elif height % 3 != 0:
        sheet_error = f"height {height} is not divisible by 3"
    else:
        tile_width = width // 2
        tile_height = height // 3
        is_sheet = True
    return ImageInfo(path, width, height, mode, is_sheet, tile_width, tile_height, sheet_error)


def crop_sheet_slot(path: Path, slot: int) -> Image.Image:
    with Image.open(path).convert("RGB") as image:
        width, height = image.size
        if width % 2 != 0 or height % 3 != 0:
            raise ValueError(f"Invalid 3x2 sheet dimensions for {path}: {width}x{height}")
        tile_width = width // 2
        tile_height = height // 3
        row = slot // 2
        col = slot % 2
        return image.crop((col * tile_width, row * tile_height, (col + 1) * tile_width, (row + 1) * tile_height))


def image_to_tensor(image: Image.Image):
    import torch

    if image.mode != "RGB":
        image = image.convert("RGB")
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(image.height, image.width, 3).permute(2, 0, 1).float()
    return data.div(127.5).sub(1.0).unsqueeze(0)


def resize_to_match(generated: Image.Image, ground_truth: Image.Image) -> tuple[Image.Image, Image.Image, str]:
    if generated.size == ground_truth.size:
        return generated, ground_truth, "none"
    return generated, ground_truth.resize(generated.size, Image.Resampling.BICUBIC), "gt_resized_to_generated"


def infer_slot_from_name(path: Path) -> tuple[int | None, int | None]:
    text = path.stem.lower()
    slot_match = re.search(r"(?:slot|view|tile)[_-]?([0-5])(?:\D|$)", text)
    if slot_match:
        slot = int(slot_match.group(1))
        return slot, SLOT_AZIMUTHS[slot]
    angle_match = re.search(r"(?<![0-9])0?([0-9]{2,3})(?:deg|degree|az|el|_|-|$)", text)
    if angle_match:
        angle = int(angle_match.group(1)) % 360
        if angle in ANGLE_TO_SLOT:
            return ANGLE_TO_SLOT[angle], angle
    return None, None


def looks_like_ground_truth(path: Path) -> bool:
    text = " ".join(part.lower() for part in path.parts)
    return any(term in text for term in GT_TERMS)


def manifest_objects(manifest: dict | None) -> dict[str, dict]:
    if not isinstance(manifest, dict):
        return {}
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        return {}
    return {
        str(entry.get("object_id")): entry
        for entry in objects
        if isinstance(entry, dict) and entry.get("object_id")
    }


def public_objects_dir(study_dir: Path) -> Path:
    candidates = [study_dir / "objects", study_dir / "public_stimuli"]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if existing:
        def score(candidate: Path) -> tuple[int, int]:
            object_dirs = [path for path in candidate.iterdir() if path.is_dir() and not path.name.startswith(".")]
            generated_dirs = [
                path
                for path in object_dirs
                if (path / "results").exists() and any((path / "results").rglob("zero123plus_sheet.png"))
            ]
            return len(generated_dirs), len(object_dirs)

        return max(existing, key=score)
    raise FileNotFoundError(f"No public object directory found under {study_dir}")


def candidate_source_roots(study_dir: Path) -> list[Path]:
    roots = [
        study_dir.parent / "rag_quick_test_views",
        REPO_ROOT / "outputs" / "rag_quick_test_views",
        REPO_ROOT / "data",
        REPO_ROOT / "test_objects",
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root.exists() and root not in seen:
            unique.append(root)
            seen.add(root)
    return unique


def index_source_inputs(roots: list[Path]) -> dict[str, list[Path]]:
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        for input_path in root.rglob("input.png"):
            try:
                by_hash[sha256_file(input_path)].append(input_path.parent)
            except OSError:
                continue
    return dict(by_hash)


def match_public_objects(
    public_root: Path,
    source_index: dict[str, list[Path]],
    manifest_by_id: dict[str, dict],
    overrides: dict[str, str],
) -> dict[str, dict]:
    matches: dict[str, dict] = {}
    for object_dir in sorted((p for p in public_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        if object_dir.name.startswith("."):
            continue
        if not (object_dir / "input.png").exists() and not (object_dir / "results").exists():
            continue
        input_path = object_dir / "input.png"
        source_dirs: list[Path] = []
        object_id: str | None = overrides.get(object_dir.name)
        category: str | None = None
        if input_path.exists():
            source_dirs = source_index.get(sha256_file(input_path), [])
            if object_id is None:
                for source_dir in source_dirs:
                    if source_dir.name in manifest_by_id:
                        object_id = source_dir.name
                        break
        if object_id and object_id in manifest_by_id:
            category = manifest_by_id[object_id].get("category")
        matches[object_dir.name] = {
            "public_object_dir": object_dir,
            "input_path": input_path if input_path.exists() else None,
            "object_id": object_id,
            "category": category,
            "source_dirs": source_dirs,
        }
    return matches


def discover_generated(public_matches: dict[str, dict], answer_key: dict[str, dict[str, str]]) -> list[GeneratedImage]:
    generated: list[GeneratedImage] = []
    for public_key, info in public_matches.items():
        object_dir = info["public_object_dir"]
        object_id = info.get("object_id")
        result_root = object_dir / "results"
        if not result_root.exists():
            continue
        for sheet_path in sorted(result_root.rglob("*")):
            if not sheet_path.is_file() or sheet_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if "model" in [part.lower() for part in sheet_path.parts]:
                continue
            rel_parts = sheet_path.relative_to(result_root).parts
            blind_result = rel_parts[0] if rel_parts else sheet_path.stem
            raw_method = answer_key.get(object_id or "", {}).get(blind_result, blind_result)
            kind = "sheet" if inspect_image(sheet_path).is_sheet else "separate_or_invalid"
            generated.append(
                GeneratedImage(
                    public_object_dir=object_dir,
                    public_object_key=public_key,
                    object_id=object_id,
                    category=info.get("category"),
                    blind_result=blind_result,
                    method=canonical_method(raw_method),
                    raw_method=raw_method,
                    path=sheet_path,
                    kind=kind,
                )
            )
    return generated


def load_meta_ground_truth(source_dir: Path, object_id: str) -> list[GroundTruthImage]:
    results: list[GroundTruthImage] = []
    meta_path = source_dir / "meta.json"
    meta = load_json(meta_path)
    if not isinstance(meta, dict):
        return results
    rendered = meta.get("rendered_views")
    if not isinstance(rendered, dict):
        return results
    for relative_name, values in rendered.items():
        if not isinstance(values, dict):
            continue
        candidate = source_dir / relative_name
        if not candidate.exists() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not looks_like_ground_truth(candidate):
            continue
        azimuth = values.get("azimuth")
        if azimuth is None:
            continue
        angle = int(round(float(azimuth))) % 360
        slot = ANGLE_TO_SLOT.get(angle)
        if slot is None:
            continue
        results.append(GroundTruthImage(object_id, slot, angle, candidate, "metadata_rendered_views", "separate"))
    return results


def discover_ground_truth_for_object(study_dir: Path, public_info: dict, object_id: str) -> list[GroundTruthImage]:
    candidates: list[GroundTruthImage] = []
    search_dirs: list[Path] = [
        public_info["public_object_dir"],
        study_dir / "ground_truth_targets" / object_id,
        study_dir / "ground_truth" / object_id,
        study_dir / "targets" / object_id,
        *public_info.get("source_dirs", []),
    ]
    for root in search_dirs:
        if not root.exists():
            continue
        for path in iter_images(root):
            if path.name.lower() == "input.png":
                continue
            if not looks_like_ground_truth(path):
                continue
            info = inspect_image(path)
            if info.is_sheet and path.name.lower() != "zero123plus_sheet.png":
                candidates.append(GroundTruthImage(object_id, None, None, path, "ground_truth_sheet", "sheet"))
                continue
            slot, angle = infer_slot_from_name(path)
            if slot is not None:
                candidates.append(GroundTruthImage(object_id, slot, angle, path, "ground_truth_separate", "separate"))
        candidates.extend(load_meta_ground_truth(root, object_id))

    # Do not treat the generated result sheets as ground truth even though they contain target-like views.
    safe_candidates = [
        candidate
        for candidate in candidates
        if "results" not in [part.lower() for part in candidate.path.parts]
    ]
    by_key: dict[tuple[str, int | None, str], GroundTruthImage] = {}
    for candidate in safe_candidates:
        key = (str(candidate.path.resolve()), candidate.slot, candidate.kind)
        by_key[key] = candidate
    return sorted(by_key.values(), key=lambda x: (x.kind, x.slot if x.slot is not None else -1, str(x.path)))


def choose_ground_truth_slots(gt_images: list[GroundTruthImage]) -> tuple[dict[int, GroundTruthImage], str]:
    separate = {gt.slot: gt for gt in gt_images if gt.kind == "separate" and gt.slot is not None}
    if len(separate) == 6:
        return separate, "six_separate_exact_target_views"

    sheet = next((gt for gt in gt_images if gt.kind == "sheet"), None)
    if sheet is not None:
        return {slot: sheet for slot in range(6)}, "complete_3x2_ground_truth_sheet"

    if separate:
        return separate, "partial_separate_exact_target_views"
    return {}, "none"


def build_inventory_rows(
    study_dir: Path,
    public_matches: dict[str, dict],
    generated: list[GeneratedImage],
    ground_truth: dict[str, list[GroundTruthImage]],
    manifest_by_id: dict[str, dict],
) -> list[dict]:
    generated_by_public = defaultdict(list)
    for image in generated:
        generated_by_public[image.public_object_key].append(image)

    rows: list[dict] = []
    for public_key, info in public_matches.items():
        object_id = info.get("object_id")
        object_generated = generated_by_public.get(public_key, [])
        methods = sorted({image.method for image in object_generated})
        image_infos = []
        for image in object_generated:
            inspected = inspect_image(image.path)
            image_infos.append(
                f"{image.blind_result}:{image.method}:{'sheet' if inspected.is_sheet else inspected.sheet_error}"
            )
        gt_images = ground_truth.get(object_id or "", [])
        gt_slots, gt_mode = choose_ground_truth_slots(gt_images)
        rows.append(
            {
                "public_object_key": public_key,
                "object_id": object_id or "",
                "category": info.get("category") or "",
                "input_path": rel(info["input_path"], study_dir) if info.get("input_path") else "",
                "source_dirs": ";".join(rel(path, study_dir) for path in info.get("source_dirs", [])),
                "methods": ";".join(methods),
                "generated_image_count": len(object_generated),
                "generated_inputs": ";".join(image_infos),
                "ground_truth_image_count": len(gt_images),
                "ground_truth_slots": ";".join(str(slot) for slot in sorted(gt_slots)),
                "ground_truth_mode": gt_mode,
                "ground_truth_paths": ";".join(rel(gt.path, study_dir) for gt in gt_images),
            }
        )

    present_ids = {row["object_id"] for row in rows if row["object_id"]}
    for object_id, entry in sorted(manifest_by_id.items()):
        if object_id not in present_ids:
            rows.append(
                {
                    "public_object_key": "",
                    "object_id": object_id,
                    "category": entry.get("category", ""),
                    "input_path": "",
                    "source_dirs": "",
                    "methods": "",
                    "generated_image_count": 0,
                    "generated_inputs": "",
                    "ground_truth_image_count": 0,
                    "ground_truth_slots": "",
                    "ground_truth_mode": "missing_public_object_folder",
                    "ground_truth_paths": "",
                }
            )
    return rows


def evaluate_lpips(
    generated: list[GeneratedImage],
    ground_truth: dict[str, list[GroundTruthImage]],
    device_name: str,
    net: str,
) -> list[dict]:
    import torch
    import lpips

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = lpips.LPIPS(net=net).to(device)
    model.eval()

    rows: list[dict] = []
    for generated_image in generated:
        if not generated_image.object_id:
            continue
        gt_images = ground_truth.get(generated_image.object_id, [])
        gt_slots, gt_mode = choose_ground_truth_slots(gt_images)
        if not gt_slots:
            continue
        generated_info = inspect_image(generated_image.path)
        if not generated_info.is_sheet:
            continue

        for slot in range(6):
            gt = gt_slots.get(slot)
            if gt is None:
                continue
            gen_tile = crop_sheet_slot(generated_image.path, slot)
            if gt.kind == "sheet":
                gt_tile = crop_sheet_slot(gt.path, slot)
            else:
                gt_tile = Image.open(gt.path).convert("RGB")
            gen_tile, gt_tile, resize_policy = resize_to_match(gen_tile, gt_tile)
            with torch.no_grad():
                score = model(image_to_tensor(gen_tile).to(device), image_to_tensor(gt_tile).to(device)).item()
            rows.append(
                {
                    "object_id": generated_image.object_id,
                    "public_object_key": generated_image.public_object_key,
                    "category": generated_image.category or "",
                    "blind_result": generated_image.blind_result,
                    "method": generated_image.method,
                    "raw_method": generated_image.raw_method,
                    "slot": slot,
                    "azimuth": SLOT_AZIMUTHS[slot],
                    "lpips": score,
                    "gt_mode": gt_mode,
                    "generated_path": rel(generated_image.path),
                    "ground_truth_path": rel(gt.path),
                    "resize_policy": resize_policy,
                    "net": net,
                    "device": str(device),
                }
            )
    return rows


def summarize_scores(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rows:
        columns = ["method", "mean_lpips", "std_lpips", "min_lpips", "max_lpips", "comparisons", "objects"]
        return pd.DataFrame(columns=columns), pd.DataFrame()
    frame = pd.DataFrame(rows)
    method_summary = (
        frame.groupby("method", as_index=False)
        .agg(
            mean_lpips=("lpips", "mean"),
            std_lpips=("lpips", "std"),
            min_lpips=("lpips", "min"),
            max_lpips=("lpips", "max"),
            comparisons=("lpips", "count"),
            objects=("object_id", "nunique"),
        )
        .sort_values(["mean_lpips", "method"])
    )
    object_summary = (
        frame.groupby(["object_id", "method"], as_index=False)
        .agg(mean_lpips=("lpips", "mean"), comparisons=("lpips", "count"))
        .sort_values(["object_id", "mean_lpips", "method"])
    )
    return method_summary, object_summary


def write_missing_report(
    path: Path,
    study_dir: Path,
    inventory_rows: list[dict],
    missing_generated: list[str],
    duplicate_ids: list[str],
) -> None:
    lines = [
        "LPIPS missing ground-truth report",
        "",
        f"Detected user-study folder: {study_dir}",
        "",
    ]
    if duplicate_ids:
        lines.append("Duplicate object IDs detected:")
        lines.extend(f"- {object_id}" for object_id in duplicate_ids)
        lines.append("")
    if missing_generated:
        lines.append("Manifest objects missing from public/generated object folders:")
        lines.extend(f"- {object_id}" for object_id in missing_generated)
        lines.append("")
    lines.append("Objects without complete valid LPIPS ground truth:")
    any_missing = False
    for row in inventory_rows:
        if row["generated_image_count"] == 0:
            continue
        slots = [int(slot) for slot in row["ground_truth_slots"].split(";") if slot != ""]
        missing_slots = sorted(set(range(6)) - set(slots))
        if missing_slots:
            any_missing = True
            lines.append(
                f"- public object {row['public_object_key']} / {row['object_id'] or 'unmatched'}: "
                f"missing slots {missing_slots}; detected gt mode={row['ground_truth_mode']}; "
                f"gt paths={row['ground_truth_paths'] or 'none'}"
            )
    if not any_missing:
        lines.append("- None")
    lines.append("")
    lines.append(
        "Safety note: reference/input images were not used as LPIPS ground truth unless they were explicitly "
        "identified as target/ground_truth/reference_render/rendered_views images with a matching Zero123++ slot."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_inventory(
    study_dir: Path,
    generated: list[GeneratedImage],
    inventory_rows: list[dict],
    duplicate_ids: list[str],
    missing_generated: list[str],
) -> None:
    methods = Counter(image.method for image in generated)
    image_modes = Counter(inspect_image(image.path).is_sheet for image in generated)
    print("LPIPS user-study inventory")
    print(f"- detected user-study folder: {study_dir}")
    print(f"- detected methods: {', '.join(f'{method} ({count})' for method, count in sorted(methods.items())) or 'none'}")
    objects = [row["object_id"] or f"unmatched:{row['public_object_key']}" for row in inventory_rows if row["public_object_key"]]
    print(f"- detected object IDs: {', '.join(objects) or 'none'}")
    print("- number of images per method:")
    for method, count in sorted(methods.items()):
        print(f"  {method}: {count}")
    gt_count = sum(int(row["ground_truth_image_count"]) for row in inventory_rows if row["public_object_key"])
    print(f"- detected ground-truth images: {gt_count}")
    missing_rows = [
        row
        for row in inventory_rows
        if row["public_object_key"] and row["generated_image_count"] and row["ground_truth_slots"] != "0;1;2;3;4;5"
    ]
    print(f"- missing files / invalid comparisons: {len(missing_rows)} object(s) missing one or more GT slots")
    if missing_generated:
        print(f"- manifest objects missing generated folders: {', '.join(missing_generated)}")
    print(f"- duplicate object IDs: {', '.join(duplicate_ids) if duplicate_ids else 'none'}")
    if image_modes:
        sheet_count = image_modes.get(True, 0)
        separate_count = image_modes.get(False, 0)
        print(f"- generated inputs: {sheet_count} complete 3x2 sheets, {separate_count} separate/invalid images")
    for row in inventory_rows:
        if not row["public_object_key"]:
            continue
        print(
            f"  object {row['public_object_key']} -> {row['object_id'] or 'unmatched'}: "
            f"{row['generated_image_count']} generated, gt_slots=[{row['ground_truth_slots']}], "
            f"mode={row['ground_truth_mode']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate user-study Zero123++ sheets with LPIPS.")
    parser.add_argument("--user-study-dir", default=None, help="User-study directory. If omitted, auto-detect.")
    parser.add_argument("--output-subdir", default="lpips_results")
    parser.add_argument("--net", default="alex", choices=("alex", "vgg", "squeeze"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inventory-only", action="store_true", help="Only write inventory/missing reports.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = Path.cwd()
    study_dir = find_user_study_dir(start, args.user_study_dir)
    output_dir = study_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(study_dir / "study_manifest.json")
    manifest_by_id = manifest_objects(manifest if isinstance(manifest, dict) else None)
    answer_key = parse_answer_key(study_dir / "private_answer_key" / "answer_key_private.md")
    public_root = public_objects_dir(study_dir)
    source_index = index_source_inputs(candidate_source_roots(study_dir))
    overrides = load_public_object_overrides(study_dir)
    public_matches = match_public_objects(public_root, source_index, manifest_by_id, overrides)
    generated = discover_generated(public_matches, answer_key)

    ground_truth: dict[str, list[GroundTruthImage]] = {}
    for info in public_matches.values():
        object_id = info.get("object_id")
        if object_id:
            ground_truth[object_id] = discover_ground_truth_for_object(study_dir, info, object_id)

    inventory_rows = build_inventory_rows(study_dir, public_matches, generated, ground_truth, manifest_by_id)
    id_counts = Counter(row["object_id"] for row in inventory_rows if row["object_id"] and row["public_object_key"])
    duplicate_ids = sorted(object_id for object_id, count in id_counts.items() if count > 1)
    public_ids = {row["object_id"] for row in inventory_rows if row["object_id"] and row["public_object_key"]}
    missing_generated = sorted(set(manifest_by_id) - public_ids)

    inventory_path = output_dir / "dataset_inventory.csv"
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    missing_report_path = output_dir / "missing_ground_truth_report.txt"
    write_missing_report(missing_report_path, study_dir, inventory_rows, missing_generated, duplicate_ids)
    print_inventory(study_dir, generated, inventory_rows, duplicate_ids, missing_generated)

    if args.inventory_only:
        print(f"Inventory written to {inventory_path}")
        print(f"Missing-ground-truth report written to {missing_report_path}")
        return 0

    evaluable_objects = [
        row
        for row in inventory_rows
        if row["public_object_key"] and row["generated_image_count"] and row["ground_truth_slots"]
    ]
    if not evaluable_objects:
        per_view_columns = [
            "object_id",
            "public_object_key",
            "category",
            "blind_result",
            "method",
            "raw_method",
            "slot",
            "azimuth",
            "lpips",
            "gt_mode",
            "generated_path",
            "ground_truth_path",
            "resize_policy",
            "net",
            "device",
        ]
        pd.DataFrame(columns=per_view_columns).to_csv(output_dir / "lpips_per_view.csv", index=False)
        pd.DataFrame(columns=["method", "mean_lpips", "std_lpips", "min_lpips", "max_lpips", "comparisons", "objects"]).to_csv(
            output_dir / "lpips_summary_by_method.csv", index=False
        )
        pd.DataFrame(columns=["object_id", "method", "mean_lpips", "comparisons"]).to_csv(
            output_dir / "lpips_summary_by_object_method.csv", index=False
        )
        print("No valid matching ground-truth slots were found; LPIPS score tables were written empty.")
        return 0

    rows = evaluate_lpips(generated, ground_truth, args.device, args.net)
    per_view = pd.DataFrame(rows)
    per_view_path = output_dir / "lpips_per_view.csv"
    per_view.to_csv(per_view_path, index=False)
    method_summary, object_summary = summarize_scores(rows)
    method_summary_path = output_dir / "lpips_summary_by_method.csv"
    object_summary_path = output_dir / "lpips_summary_by_object_method.csv"
    method_summary.to_csv(method_summary_path, index=False)
    object_summary.to_csv(object_summary_path, index=False)

    print(f"LPIPS per-view results written to {per_view_path}")
    print(f"LPIPS method summary written to {method_summary_path}")
    if method_summary.empty:
        print("No LPIPS comparisons were completed.")
    else:
        print("LPIPS summary by method (lower is better):")
        print(method_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
