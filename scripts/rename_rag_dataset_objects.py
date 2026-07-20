import argparse
import json
import os
from pathlib import Path


JSONL_NAMES = ("train.jsonl", "val.jsonl", "test_objaverse_heldout.jsonl")
PRESERVE_VALUE_KEYS = {"original_object_id", "original_name", "source_path"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename RAG dataset object folders to object_000001-style names and rewrite manifests."
    )
    parser.add_argument("--root", required=True, help="Dataset root containing objects/ and JSONL manifests.")
    parser.add_argument("--prefix", default="object", help="New object id prefix.")
    parser.add_argument("--digits", type=int, default=6, help="Zero padding digits for new ids.")
    parser.add_argument("--start_index", type=int, default=1, help="First numeric id.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned changes without writing anything.")
    return parser.parse_args()


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def collect_object_ids(root):
    seen = set()
    ordered = []
    for name in JSONL_NAMES:
        for record in load_jsonl(root / name):
            object_id = str(record.get("object_id", "")).strip()
            if object_id and object_id not in seen:
                seen.add(object_id)
                ordered.append(object_id)

    objects_dir = root / "objects"
    if objects_dir.exists():
        for path in sorted(item for item in objects_dir.iterdir() if item.is_dir()):
            if path.name not in seen:
                seen.add(path.name)
                ordered.append(path.name)
    return ordered


def build_mapping(object_ids, prefix, digits, start_index):
    return {
        old_id: f"{prefix}_{index:0{digits}d}"
        for index, old_id in enumerate(object_ids, start=start_index)
    }


def replace_string(value, mapping):
    for old_id, new_id in mapping.items():
        if value == old_id:
            return new_id
    result = value
    for old_id, new_id in mapping.items():
        result = result.replace(f"objects/{old_id}/", f"objects/{new_id}/")
        result = result.replace(f"objects\\{old_id}\\", f"objects\\{new_id}\\")
    return result


def replace_key(value, mapping):
    return mapping.get(value, value)


def rewrite_json(value, mapping, parent_key=None):
    if isinstance(value, str):
        if parent_key in PRESERVE_VALUE_KEYS:
            return value
        return replace_string(value, mapping)
    if isinstance(value, list):
        return [rewrite_json(item, mapping, parent_key=parent_key) for item in value]
    if isinstance(value, dict):
        rewritten = {}
        for key, item in value.items():
            new_key = replace_key(str(key), mapping)
            rewritten[new_key] = rewrite_json(item, mapping, parent_key=new_key)
        return rewritten
    return value


def add_original_ids_to_record(record, mapping):
    old_id = record.get("object_id")
    if old_id in mapping:
        metadata = dict(record.get("metadata", {}))
        metadata.setdefault("original_object_id", old_id)
        if metadata.get("name") == old_id:
            metadata["name"] = mapping[old_id]
        else:
            metadata.setdefault("original_name", metadata.get("name", old_id))
        record["metadata"] = metadata
    return record


def rewrite_manifest(path, mapping, dry_run):
    rows = load_jsonl(path)
    if not rows:
        return 0
    rows = [rewrite_json(add_original_ids_to_record(row, mapping), mapping) for row in rows]
    if not dry_run:
        write_jsonl(path, rows)
    return len(rows)


def rewrite_json_file(path, mapping, dry_run):
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    object_id = data.get("object_id") if isinstance(data, dict) else None
    if object_id in mapping:
        data.setdefault("original_object_id", object_id)
        if data.get("name") == object_id:
            data["name"] = mapping[object_id]
    data = rewrite_json(data, mapping)
    if not dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return True


def rename_directories(root, mapping, dry_run):
    objects_dir = root / "objects"
    if not objects_dir.exists():
        raise FileNotFoundError(f"Missing objects directory: {objects_dir}")

    existing_targets = [
        new_id for old_id, new_id in mapping.items()
        if old_id != new_id and (objects_dir / new_id).exists() and new_id not in mapping
    ]
    if existing_targets:
        raise FileExistsError(f"Target folders already exist and are not part of this rename: {existing_targets[:5]}")

    temp_suffix = f".renaming_tmp_{os.getpid()}"
    temp_mapping = {
        old_id: f"{new_id}{temp_suffix}"
        for old_id, new_id in mapping.items()
        if old_id != new_id and (objects_dir / old_id).exists()
    }

    if dry_run:
        return len(temp_mapping)

    for old_id, temp_id in temp_mapping.items():
        (objects_dir / old_id).rename(objects_dir / temp_id)
    for old_id, temp_id in temp_mapping.items():
        (objects_dir / temp_id).rename(objects_dir / mapping[old_id])
    return len(temp_mapping)


def write_mapping(root, mapping, dry_run):
    rows = [
        {"old_object_id": old_id, "new_object_id": new_id}
        for old_id, new_id in mapping.items()
    ]
    if not dry_run:
        (root / "object_rename_map.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def main():
    args = parse_args()
    root = Path(args.root)
    object_ids = collect_object_ids(root)
    mapping = build_mapping(object_ids, args.prefix, args.digits, args.start_index)

    print(f"[rename-rag-objects] root: {root}")
    print(f"[rename-rag-objects] objects found: {len(object_ids)}")
    print(f"[rename-rag-objects] dry_run: {args.dry_run}")
    for old_id, new_id in list(mapping.items())[:10]:
        print(f"[rename-rag-objects] {old_id} -> {new_id}")
    if len(mapping) > 10:
        print(f"[rename-rag-objects] ... {len(mapping) - 10} more")

    for name in JSONL_NAMES:
        count = rewrite_manifest(root / name, mapping, args.dry_run)
        if count:
            print(f"[rename-rag-objects] rewritten {name}: {count} records")

    split_report = root / "split_report.json"
    if rewrite_json_file(split_report, mapping, args.dry_run):
        print("[rename-rag-objects] rewritten split_report.json")

    renamed_dirs = rename_directories(root, mapping, args.dry_run)

    objects_dir = root / "objects"
    for old_id, new_id in mapping.items():
        meta_path = objects_dir / (new_id if not args.dry_run else old_id) / "meta.json"
        rewrite_json_file(meta_path, mapping, args.dry_run)

    write_mapping(root, mapping, args.dry_run)
    print(f"[rename-rag-objects] folders renamed: {renamed_dirs}")
    if not args.dry_run:
        print(f"[rename-rag-objects] mapping written: {root / 'object_rename_map.json'}")


if __name__ == "__main__":
    main()
