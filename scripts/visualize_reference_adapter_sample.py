import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


TARGET_ORDER = [
    ("target_030.png", "target 30"),
    ("target_090.png", "target 90"),
    ("target_150.png", "target 150"),
    ("target_210.png", "target 210"),
    ("target_270.png", "target 270"),
    ("target_330.png", "target 330"),
]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="data/reference_zero123plus_tiny")
    parser.add_argument("--object_id", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--thumb_size", type=int, default=160)
    return parser.parse_args()


def open_thumb(path, size):
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def labeled_tile(path, label, size):
    tile = Image.new("RGB", (size, size + 24), "white")
    tile.paste(open_thumb(path, size), (0, 24))
    draw = ImageDraw.Draw(tile)
    draw.text((4, 4), label, fill=(0, 0, 0))
    return tile


def load_record(root, object_id):
    manifest = root / "train.jsonl"
    with open(manifest, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if object_id:
        for record in records:
            if record["object_id"] == object_id:
                return record
        raise ValueError(f"Object id not found in manifest: {object_id}")
    return records[0]


def main():
    args = parse_args()
    root = Path(args.root_dir)
    record = load_record(root, args.object_id)
    object_dir = root / record["object_id"]
    output = Path(args.output) if args.output else object_dir / "preview_grid.png"
    size = args.thumb_size

    rows = [
        [labeled_tile(object_dir / "cond.png", "cond 0", size)],
        [labeled_tile(object_dir / name, label, size) for name, label in TARGET_ORDER],
        [
            labeled_tile(root / ref_path, f"ref {int(azimuth):03d} el {float(elevation):+.0f}", size)
            for ref_path, azimuth, elevation in zip(
                record["ref_imgs"],
                record.get("ref_azimuths", []),
                record.get("ref_elevations", []),
            )
        ],
    ]

    cols = 6
    tile_h = size + 24
    grid = Image.new("RGB", (cols * size, len(rows) * tile_h), "white")
    for row_idx, row_tiles in enumerate(rows):
        for col_idx, tile in enumerate(row_tiles):
            grid.paste(tile, (col_idx * size, row_idx * tile_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    print(f"[visualize] wrote {output}")


if __name__ == "__main__":
    main()
