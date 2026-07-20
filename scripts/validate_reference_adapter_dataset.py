import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.objaverse_zero123plus import ManifestReferenceZero123PlusData


TARGET_ORDER = ["30", "90", "150", "210", "270", "330"]
TARGET_AZIMUTHS = [30, 90, 150, 210, 270, 330]
REQUIRED_FILENAMES = [
    "cond.png",
    "target_030.png",
    "target_090.png",
    "target_150.png",
    "target_210.png",
    "target_270.png",
    "target_330.png",
    "meta.json",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="data/reference_zero123plus_tiny")
    parser.add_argument("--manifest", default="train.jsonl")
    return parser.parse_args()


def load_records(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_image(path, expected_size=None):
    with Image.open(path) as image:
        image.load()
        if expected_size is not None and image.size != expected_size:
            raise ValueError(f"Size mismatch for {path}: {image.size} != {expected_size}")
        return image.size


def main():
    args = parse_args()
    root = Path(args.root_dir)
    manifest_path = root / args.manifest
    records = load_records(manifest_path)
    if not records:
        raise ValueError(f"No records found in {manifest_path}")

    for record in records:
        if list(record["target_imgs"].keys()) != TARGET_ORDER:
            raise ValueError(f"{record['object_id']} target order is not {TARGET_ORDER}")
        if record.get("target_azimuths") != TARGET_AZIMUTHS:
            raise ValueError(f"{record['object_id']} target_azimuths must be {TARGET_AZIMUTHS}")
        if len(record.get("target_elevations", [])) != 6:
            raise ValueError(f"{record['object_id']} target_elevations must have length 6")
        expected_size = check_image(root / record["cond_img"])
        for azimuth in TARGET_ORDER:
            check_image(root / record["target_imgs"][azimuth], expected_size)
        for ref_path in record["ref_imgs"]:
            check_image(root / ref_path, expected_size)
        if len(record.get("ref_azimuths", [])) != len(record.get("ref_imgs", [])):
            raise ValueError(f"{record['object_id']} ref_azimuths length must match ref_imgs")
        if len(record.get("ref_elevations", [])) != len(record.get("ref_imgs", [])):
            raise ValueError(f"{record['object_id']} ref_elevations length must match ref_imgs")
        meta_path = root / record["object_id"] / "meta.json"
        if not meta_path.exists():
            raise ValueError(f"Missing meta.json: {meta_path}")
        for filename in REQUIRED_FILENAMES:
            required_path = root / record["object_id"] / filename
            if not required_path.exists():
                raise ValueError(f"Missing required file: {required_path}")
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if not meta.get("validation_passed", False):
            raise ValueError(f"{record['object_id']} meta validation failed: {meta.get('validation_reason')}")
        rendered_views = meta.get("rendered_views", {})
        for filename in ["cond.png", *REQUIRED_FILENAMES[1:-1], *[Path(path).name for path in record["ref_imgs"]]]:
            pose = rendered_views.get(filename)
            if not pose or "azimuth" not in pose or "elevation" not in pose:
                raise ValueError(f"{record['object_id']} meta missing rendered pose for {filename}")

    dataset = ManifestReferenceZero123PlusData(root_dir=str(root), manifest_fname=args.manifest, validation=False)
    sample = dataset[0]
    if sample["ref_slot_weights"].ndim != 2 or sample["ref_slot_weights"].shape[1] != 6:
        raise ValueError(f"ref_slot_weights shape is wrong: {sample['ref_slot_weights'].shape}")

    print(f"[validate] records: {len(records)}")
    print(f"[validate] image size: {expected_size}")
    print(f"[validate] ref_slot_weights shape: {tuple(sample['ref_slot_weights'].shape)}")
    print("[validate] OK")


if __name__ == "__main__":
    main()
