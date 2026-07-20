import argparse
import json
from pathlib import Path


def load_bad_ids(path):
    bad_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.split("#", 1)[0].strip()
            if value:
                bad_ids.add(value)
    return bad_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--bad_list", required=True)
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()

    bad_ids = load_bad_ids(args.bad_list)
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    kept = []
    excluded = []
    with open(input_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = record.get("object_id")
            if not object_id:
                raise ValueError(f"Missing object_id at {input_path}:{line_number}")
            if object_id in bad_ids:
                excluded.append(object_id)
            else:
                kept.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record) + "\n")

    missing = sorted(bad_ids - set(excluded))
    print(f"[filter] input={len(kept) + len(excluded)} kept={len(kept)} excluded={len(excluded)}")
    print(f"[filter] wrote {output_path}")
    if missing:
        print(f"[filter:warning] bad IDs not present in input manifest: {missing}")


if __name__ == "__main__":
    main()
