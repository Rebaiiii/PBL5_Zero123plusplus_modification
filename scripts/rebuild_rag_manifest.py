import argparse
import json
import shutil
from pathlib import Path


TARGET_AZIMUTHS = [30, 90, 150, 210, 270, 330]


def build_record(dataset_dir, object_dir, accept_complete_validation_failures=False):
    meta_path = object_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = [object_dir / "cond.png", meta_path]
    required.extend(object_dir / f"target_{azimuth:03d}.png" for azimuth in TARGET_AZIMUTHS)
    missing = [path.name for path in required if not path.exists()]
    ref_files = sorted(object_dir.glob("ref_*.png"))
    if len(ref_files) < 3:
        missing.append(f"expected at least 3 ref PNGs, found {len(ref_files)}")
    if missing:
        raise ValueError(", ".join(missing))

    if meta.get("validation_passed") is not True:
        if not accept_complete_validation_failures:
            raise ValueError(f"meta validation failed: {meta.get('validation_reason')}")
        meta["manual_training_include"] = True
        meta["original_validation_reason"] = meta.get("validation_reason")
        meta["validation_passed"] = True
        meta["validation_reason"] = "manually accepted for training after visual review"
        meta["status"] = "ok"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[manifest:manual-accept] {object_dir.name}")

    rendered_views = meta.get("rendered_views", {})
    ref_azimuths = []
    ref_elevations = []
    for ref_file in ref_files[:3]:
        pose = rendered_views.get(ref_file.name, {})
        if "azimuth" not in pose or "elevation" not in pose:
            raise ValueError(f"missing pose metadata for {ref_file.name}")
        ref_azimuths.append(pose["azimuth"])
        ref_elevations.append(pose["elevation"])

    object_id = object_dir.name
    return {
        "object_id": object_id,
        "cond_img": f"{object_id}/cond.png",
        "target_azimuths": meta.get("target_azimuths", TARGET_AZIMUTHS),
        "target_elevations": meta["target_elevations"],
        "target_imgs": {
            str(azimuth): f"{object_id}/target_{azimuth:03d}.png"
            for azimuth in TARGET_AZIMUTHS
        },
        "ref_imgs": [f"{object_id}/{path.name}" for path in ref_files[:3]],
        "ref_view_labels": ["unknown"] * 3,
        "ref_azimuths": ref_azimuths,
        "ref_elevations": ref_elevations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", default="train.jsonl")
    parser.add_argument("--accept_complete_validation_failures", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / args.manifest
    records = []
    skipped = []
    for object_dir in sorted(dataset_dir.glob("object_*")):
        if not object_dir.is_dir():
            continue
        try:
            records.append(build_record(
                dataset_dir,
                object_dir,
                accept_complete_validation_failures=args.accept_complete_validation_failures,
            ))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            skipped.append((object_dir.name, str(exc)))

    if manifest_path.exists():
        backup_path = manifest_path.with_suffix(".before_rebuild.jsonl")
        shutil.copy2(manifest_path, backup_path)
        print(f"[manifest] backup: {backup_path}")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"[manifest] wrote {manifest_path} with {len(records)} complete object(s)")
    for object_id, reason in skipped:
        print(f"[manifest:skip] {object_id}: {reason}")


if __name__ == "__main__":
    main()
