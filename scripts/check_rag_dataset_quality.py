import argparse
import json
import math
from pathlib import Path

from PIL import Image

from postprocess_rag_render import image_report


TARGET_KEYS = ["30", "90", "150", "210", "270", "330"]


def circular_distance(a, b):
    difference = abs(float(a) - float(b)) % 360.0
    return min(difference, 360.0 - difference)


def pose_weights(azimuth, elevation, target_azimuths, target_elevations):
    weights = []
    for target_azimuth, target_elevation in zip(target_azimuths, target_elevations):
        azimuth_distance = circular_distance(azimuth, target_azimuth)
        elevation_distance = abs(float(elevation) - float(target_elevation))
        value = math.exp(
            -(azimuth_distance ** 2) / (2 * 45.0 ** 2)
            -(elevation_distance ** 2) / (2 * 25.0 ** 2)
        )
        weights.append(max(0.05, value))
    return weights


def inspect_image(path, expected_size, azimuth=None):
    issues = []
    try:
        with Image.open(path) as image:
            image.load()
            report = image_report(image)
            report["format"] = image.format
    except Exception as exc:
        return None, [f"cannot open image: {exc}"]

    if tuple(report["resolution"]) != expected_size:
        issues.append(f"resolution={report['resolution']} expected={list(expected_size)}")
    if report["format"] != "PNG":
        issues.append(f"format={report['format']} expected=PNG")
    if report["mode"] != "RGB":
        issues.append(f"mode={report['mode']} expected=RGB")
    if report["has_alpha"]:
        issues.append("contains alpha")
    if report["bbox"] is None:
        issues.append("no foreground detected")
    else:
        if report["bbox_occupancy"] < 0.03:
            issues.append(f"mostly white/object too small occupancy={report['bbox_occupancy']:.4f}")
        if max(report["bbox_width_ratio"], report["bbox_height_ratio"]) < 0.50:
            issues.append(
                "object fills less than 50% of both dimensions "
                f"({report['bbox_width_ratio']:.1%}x{report['bbox_height_ratio']:.1%})"
            )
        if report["edge_variance"] < 25.0:
            issues.append(f"possibly blurry edge_variance={report['edge_variance']:.2f}")
        if azimuth is not None and 60 <= float(azimuth) % 360 <= 300:
            if report["foreground_luminance"] < 0.10:
                issues.append(
                    "side/back view may be too dark "
                    f"luminance={report['foreground_luminance']:.3f}"
                )
    return report, issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", default="train.jsonl")
    parser.add_argument("--expected_size", type=int, default=320)
    parser.add_argument("--output_report", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / args.manifest
    output_path = Path(args.output_report) if args.output_report else dataset_dir / "dataset_quality_report.json"
    records = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                record = json.loads(line)
                record["_line_number"] = line_number
                records.append(record)

    report = {"dataset_dir": str(dataset_dir.resolve()), "manifest": str(manifest_path), "objects": []}
    issue_count = 0
    for record in records:
        object_id = record.get("object_id", f"line_{record['_line_number']}")
        object_report = {"object_id": object_id, "issues": [], "images": {}}
        target_azimuths = record.get("target_azimuths", [30, 90, 150, 210, 270, 330])
        target_elevations = record.get("target_elevations")
        target_images = record.get("target_imgs", {})
        if list(target_images.keys()) != TARGET_KEYS:
            object_report["issues"].append(
                f"target order/keys must be {TARGET_KEYS}, got {list(target_images.keys())}"
            )

        image_items = [("cond", record.get("cond_img"), 0)]
        image_items.extend(
            (f"target_{key}", target_images.get(key), float(key)) for key in TARGET_KEYS
        )
        ref_images = record.get("ref_imgs", [])
        ref_azimuths = record.get("ref_azimuths")
        ref_elevations = record.get("ref_elevations")
        if not ref_images:
            object_report["issues"].append("missing reference image list")
        if ref_azimuths is None or ref_elevations is None:
            object_report["issues"].append("references missing azimuth/elevation metadata")
        elif len(ref_images) != len(ref_azimuths) or len(ref_images) != len(ref_elevations):
            object_report["issues"].append("reference image/azimuth/elevation lengths do not match")
        for index, ref_path in enumerate(ref_images):
            azimuth = ref_azimuths[index] if ref_azimuths and index < len(ref_azimuths) else None
            image_items.append((f"ref_{index}", ref_path, azimuth))

        for label, relative_path, azimuth in image_items:
            if not relative_path:
                object_report["issues"].append(f"{label}: missing path")
                continue
            path = dataset_dir / relative_path
            if not path.exists():
                object_report["issues"].append(f"{label}: missing file {relative_path}")
                continue
            image_metrics, image_issues = inspect_image(
                path, (args.expected_size, args.expected_size), azimuth=azimuth
            )
            object_report["images"][label] = image_metrics
            object_report["issues"].extend(f"{label}: {issue}" for issue in image_issues)

        if ref_azimuths and ref_elevations and target_elevations:
            slot_weights = [
                pose_weights(azimuth, elevation, target_azimuths, target_elevations)
                for azimuth, elevation in zip(ref_azimuths, ref_elevations)
            ]
            object_report["ref_slot_weights"] = slot_weights
            for index, weights in enumerate(slot_weights):
                if max(weights) - min(weights) < 0.02:
                    object_report["issues"].append(
                        f"ref_{index}: ref_slot_weights are nearly uniform {weights}"
                    )

        issue_count += len(object_report["issues"])
        report["objects"].append(object_report)
        status = "OK" if not object_report["issues"] else f"WARN ({len(object_report['issues'])})"
        print(f"[quality] {object_id}: {status}")
        for issue in object_report["issues"]:
            print(f"  - {issue}")

    report["summary"] = {
        "objects": len(records),
        "objects_with_issues": sum(bool(item["issues"]) for item in report["objects"]),
        "total_issues": issue_count,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[quality] summary={report['summary']}")
    print(f"[quality] wrote {output_path}")


if __name__ == "__main__":
    main()
