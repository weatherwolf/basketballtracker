"""
reverse_label.py

Reverses the label (goal -> miss or miss -> goal) for a specific shot.

Usage:
    python dev/reverse_label.py --batch live_20260321_161036 --shot 4
    python dev/reverse_label.py --batch pending_115665147 --shot frame_test1_shot042
"""

import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT, LABELS_CSV as CSV_PATH, LABELS_JSON as JSON_PATH

REVERSE = {"goal": "miss", "miss": "goal"}


def match_shot(dataset_name: str, shot: str) -> bool:
    """Match by full dataset_name or by trailing shot number."""
    if dataset_name == shot:
        return True
    # allow passing bare number, e.g. "4" matches "shot_000004"
    try:
        return dataset_name.endswith(f"_{int(shot):06d}")
    except ValueError:
        return False


def flip_label(batch_id: str, shot: str) -> bool:
    """Reverse the label for a shot. Returns True if successful, False if not found."""
    rows = []
    fieldnames = None
    found = False
    old_label = None
    new_label = None

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            parts = Path(row["rel_shot_dir"]).parts
            b     = next((p for p in parts if p.startswith(("pending_", "live_"))), "unknown")
            if b == batch_id and match_shot(row["dataset_name"], shot):
                if row["label"] not in REVERSE:
                    print(f"  Cannot reverse label '{row['label']}' — only goal/miss supported.")
                    return False
                old_label = row["label"]
                new_label = REVERSE[old_label]
                row["label"] = new_label
                found = True
            rows.append(row)

    if not found:
        print(f"  No shot found for batch={batch_id} shot={shot}")
        return False

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    data  = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    items = data.get("items", [])
    for item in items:
        parts = Path(item.get("rel_shot_dir", "")).parts
        b     = next((p for p in parts if p.startswith(("pending_", "live_"))), "unknown")
        if b == batch_id and match_shot(item.get("dataset_name", ""), shot):
            item["label"] = new_label
    JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"  {batch_id} / {shot}:  {old_label} → {new_label}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Reverse goal/miss label for a specific shot.")
    ap.add_argument("--batch", required=True, metavar="BATCH_ID")
    ap.add_argument("--shot",  required=True, metavar="SHOT",
                    help="Shot dataset_name or bare number (e.g. 4 or shot_000004)")
    args = ap.parse_args()
    flip_label(args.batch, args.shot)


if __name__ == "__main__":
    main()
