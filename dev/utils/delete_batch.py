"""
delete_batch.py

Removes all data associated with a specific batch from the pipeline.

Deletes:
  - Rows in data/shot_labels.csv and data/shot_labels.json
  - data/ball_tracking/<batch_id>_*.csv
  - data/ball_tracking_normalized/<batch_id>_*.csv
  - work/runs/<batch_id>/
  - media/exports/<batch_id>/
  - assets/hoop_ellipses/<batch_id>/

Usage:
    python dev/delete_batch.py --batch pending_115665147
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT, LABELS_CSV as CSV_PATH, LABELS_JSON as JSON_PATH, TRACKING_DIR, NORMALIZED_DIR, RUNS_DIR, EXPORTS_DIR, ELLIPSES_DIR


def find_targets(batch_id: str) -> dict:
    """Collect everything that would be deleted."""
    targets = {
        "csv_rows":    [],
        "tracking":    [],
        "normalized":  [],
        "folders":     [],
    }

    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if batch_id in row.get("rel_shot_dir", ""):
                    targets["csv_rows"].append(row["dataset_name"])

    for d in [TRACKING_DIR, NORMALIZED_DIR]:
        if d.exists():
            found = sorted(d.glob(f"{batch_id}_*.csv"))
            if d == TRACKING_DIR:
                targets["tracking"] = found
            else:
                targets["normalized"] = found

    for folder in [RUNS_DIR / batch_id, EXPORTS_DIR / batch_id, ELLIPSES_DIR / batch_id]:
        if folder.exists():
            targets["folders"].append(folder)

    return targets


def print_plan(batch_id: str, targets: dict) -> None:
    print(f"\nBatch: {batch_id}\n")

    if targets["csv_rows"]:
        print(f"  shot_labels.csv/json — {len(targets['csv_rows'])} rows:")
        for name in targets["csv_rows"]:
            print(f"    {name}")
    else:
        print("  shot_labels.csv/json — no rows found")

    if targets["tracking"]:
        print(f"\n  ball_tracking — {len(targets['tracking'])} files:")
        for p in targets["tracking"]:
            print(f"    {p.name}")

    if targets["normalized"]:
        print(f"\n  ball_tracking_normalized — {len(targets['normalized'])} files:")
        for p in targets["normalized"]:
            print(f"    {p.name}")

    if targets["folders"]:
        print(f"\n  folders:")
        for p in targets["folders"]:
            print(f"    {p.relative_to(REPO_ROOT)}")

    total = (len(targets["csv_rows"]) + len(targets["tracking"]) +
             len(targets["normalized"]) + len(targets["folders"]))
    if total == 0:
        print("\n  Nothing found for this batch.")


def delete(batch_id: str, targets: dict) -> None:
    # Remove rows from CSV
    if CSV_PATH.exists() and targets["csv_rows"]:
        rows = []
        fieldnames = None
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if batch_id not in row.get("rel_shot_dir", ""):
                    rows.append(row)
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Removed {len(targets['csv_rows'])} rows from shot_labels.csv")

    # Remove rows from JSON
    if JSON_PATH.exists() and targets["csv_rows"]:
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        items = data.get("items", [])
        kept = [it for it in items if batch_id not in it.get("rel_shot_dir", "")]
        data["items"] = kept
        JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  Removed {len(items) - len(kept)} entries from shot_labels.json")

    # Delete tracking CSVs
    for p in targets["tracking"] + targets["normalized"]:
        p.unlink()
        print(f"  Deleted {p.relative_to(REPO_ROOT)}")

    # Delete folders
    for folder in targets["folders"]:
        shutil.rmtree(folder)
        print(f"  Deleted {folder.relative_to(REPO_ROOT)}/")

    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(description="Delete all data for a specific batch.")
    ap.add_argument("--batch", required=True, metavar="BATCH_ID",
                    help="Batch id to delete (e.g. pending_115665147)")
    ap.add_argument("--yes", action="store_true",
                    help="Skip confirmation prompt")
    args = ap.parse_args()

    batch_id = args.batch
    targets  = find_targets(batch_id)

    print_plan(batch_id, targets)

    total = (len(targets["csv_rows"]) + len(targets["tracking"]) +
             len(targets["normalized"]) + len(targets["folders"]))
    if total == 0:
        return

    if not args.yes:
        print()
        confirm = input("Delete all of the above? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    print()
    delete(batch_id, targets)


if __name__ == "__main__":
    main()
