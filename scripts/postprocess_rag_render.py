import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat


def foreground_bbox(image, threshold=8):
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    red, green, blue = ImageChops.difference(rgb, background).split()
    difference = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    return difference.point(lambda value: 255 if value > threshold else 0).getbbox()


def image_report(image):
    bbox = foreground_bbox(image)
    width, height = image.size
    if bbox is None:
        bbox_width = bbox_height = 0
        occupancy = 0.0
        foreground_luminance = 0.0
        center_offset = 1.0
    else:
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        occupancy = (bbox_width * bbox_height) / float(width * height)
        bbox_center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        center_offset = (
            ((bbox_center[0] / width) - 0.5) ** 2
            + ((bbox_center[1] / height) - 0.5) ** 2
        ) ** 0.5
        foreground_luminance = ImageStat.Stat(
            image.convert("L").crop(bbox)
        ).mean[0] / 255.0

    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_stats = ImageStat.Stat(edges)
    sharpness = edge_stats.var[0]
    corner_size = min(12, width, height)
    corners = [
        image.crop((0, 0, corner_size, corner_size)),
        image.crop((width - corner_size, 0, width, corner_size)),
        image.crop((0, height - corner_size, corner_size, height)),
        image.crop((width - corner_size, height - corner_size, width, height)),
    ]
    corner_mean = sum(sum(ImageStat.Stat(crop.convert("RGB")).mean) / 3.0 for crop in corners) / len(corners)
    return {
        "resolution": [width, height],
        "mode": image.mode,
        "has_alpha": "A" in image.getbands(),
        "bbox": list(bbox) if bbox is not None else None,
        "bbox_width_ratio": bbox_width / float(width),
        "bbox_height_ratio": bbox_height / float(height),
        "bbox_occupancy": occupancy,
        "center_offset": center_offset,
        "foreground_luminance": foreground_luminance,
        "edge_variance": sharpness,
        "corner_white": corner_mean >= 245.0,
    }


def adaptive_foreground_resize(image, output_size, min_fill_ratio=0.70, target_fill_ratio=0.78):
    """Resize only undersized foregrounds by cropping around the object on white."""
    image = image.convert("RGB")
    bbox = foreground_bbox(image)
    if bbox is None:
        return image.resize((output_size, output_size), Image.Resampling.LANCZOS), {
            "adaptive_rescale_applied": False,
            "adaptive_rescale_reason": "no_foreground",
        }

    width, height = image.size
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    fill_ratio = max(bbox_width / float(width), bbox_height / float(height))
    if fill_ratio >= min_fill_ratio:
        resized = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
        return resized, {
            "adaptive_rescale_applied": False,
            "adaptive_rescale_reason": "already_large_enough",
            "source_fill_ratio": fill_ratio,
        }

    target_fill_ratio = max(min(float(target_fill_ratio), 0.95), 0.05)
    crop_size = int(round(max(bbox_width, bbox_height) / target_fill_ratio))
    crop_size = max(crop_size, max(bbox_width, bbox_height), 1)
    center_x = (bbox[0] + bbox[2]) * 0.5
    center_y = (bbox[1] + bbox[3]) * 0.5
    left = int(round(center_x - crop_size * 0.5))
    top = int(round(center_y - crop_size * 0.5))
    right = left + crop_size
    bottom = top + crop_size

    canvas = Image.new("RGB", (crop_size, crop_size), "white")
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(width, right)
    src_bottom = min(height, bottom)
    dst_left = src_left - left
    dst_top = src_top - top
    if src_right > src_left and src_bottom > src_top:
        canvas.paste(
            image.crop((src_left, src_top, src_right, src_bottom)),
            (dst_left, dst_top),
        )
    resized = canvas.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return resized, {
        "adaptive_rescale_applied": True,
        "adaptive_rescale_reason": "foreground_too_small",
        "source_fill_ratio": fill_ratio,
        "target_fill_ratio": target_fill_ratio,
        "crop_box": [left, top, right, bottom],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--adaptive_rescale", action="store_true")
    parser.add_argument("--min_fill_ratio", type=float, default=0.70)
    parser.add_argument("--target_fill_ratio", type=float, default=0.78)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    with Image.open(input_path) as source:
        source = source.convert("RGBA")
        white = Image.new("RGBA", source.size, (255, 255, 255, 255))
        white.alpha_composite(source)
        rgb = white.convert("RGB")
        if args.adaptive_rescale:
            rgb, adaptive_report = adaptive_foreground_resize(
                rgb,
                args.size,
                min_fill_ratio=args.min_fill_ratio,
                target_fill_ratio=args.target_fill_ratio,
            )
        elif rgb.size != (args.size, args.size):
            rgb = rgb.resize((args.size, args.size), Image.Resampling.LANCZOS)
            adaptive_report = {"adaptive_rescale_applied": False, "adaptive_rescale_reason": "disabled"}
        else:
            adaptive_report = {"adaptive_rescale_applied": False, "adaptive_rescale_reason": "disabled"}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"optimize": True}
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": 95, "subsampling": 0})
        rgb.save(output_path, **save_kwargs)

    report = image_report(rgb)
    report.update(adaptive_report)
    report["source_resolution"] = list(source.size)
    report["resampling"] = "Lanczos"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
