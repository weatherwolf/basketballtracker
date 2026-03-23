"""
delete_clip.py

Removes a single clip from the dataset:
  - Entry in shot_labels.json / shot_labels.csv
  - Shot frames folder
  - Normalized tracking CSV
  - Export mp4
  - Preview mp4 (if present)
  - Per-shot ellipse file (if present)

Usage (from repo root):
    python dev/delete_clip.py --batch <batch_id> --shot <dataset_name_or_shot_number>

Examples:
    python dev/delete_clip.py --batch live_20260321_181637 --shot shot_000004
    python dev/delete_clip.py --batch pending_629232164 --shot 018
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT, LABELS_JSON, LABELS_CSV, NORMALIZED_DIR, LABEL_CSV_FIELDS


def find_entry(items: list, batch_id: str, shot_arg: str) -> dict | None:
    """Find a label entry by batch_id + dataset_name or shot number suffix."""
    for item in items:
        parts    = Path(item.get("rel_shot_dir", "")).parts
        item_batch = next((p for p in parts if p.startswith(("pending_", "live_"))), None)
        if item_batch != batch_id:
            continue
        dataset_name = item.get("dataset_name", "")
        # match exact dataset_name (e.g. "shot_000004") or numeric suffix (e.g. "018")
        suffix = dataset_name.split("_shot")[-1] if "_shot" in dataset_name else dataset_name
        if dataset_name == shot_arg or suffix == shot_arg:
            return item
    return None


def collect_paths(entry: dict) -> list[tuple[str, Path]]:
    """Return list of (description, path) for everything owned by this clip."""
    paths = []

    # frames folder
    rel_shot_dir = entry.get("rel_shot_dir", "")
    if rel_shot_dir:
        paths.append(("frames folder", REPO_ROOT / rel_shot_dir))

    # normalized tracking CSV
    dataset_name = entry.get("dataset_name", "")
    parts        = Path(rel_shot_dir).parts
    batch_id     = next((p for p in parts if p.startswith(("pending_", "live_"))), None)
    if batch_id and dataset_name:
        paths.append(("normalized CSV", NORMALIZED_DIR / f"{batch_id}_{dataset_name}.csv"))

    # export mp4
    rel_export = entry.get("rel_export_mp4", "")
    if rel_export:
        paths.append(("export mp4", REPO_ROOT / rel_export))

    # preview mp4
    rel_preview = entry.get("rel_preview_mp4", "")
    if rel_preview:
        paths.append(("preview mp4", REPO_ROOT / rel_preview))

    # per-shot ellipse
    ellipse_meta = entry.get("ellipse_meta", "")
    if ellipse_meta:
        paths.append(("ellipse file", REPO_ROOT / ellipse_meta))

    return paths


def rewrite_labels(items: list, removed_rel_shot_dir: str) -> None:
    remaining = [i for i in items if i.get("rel_shot_dir") != removed_rel_shot_dir]

    data = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    data["items"] = remaining
    LABELS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(remaining)


def delete_clip(batch_id: str, shot_arg: str) -> bool:
    """Delete a clip and remove it from shot_labels. Returns True if deleted, False if not found."""
    if not LABELS_JSON.exists():
        print(f"shot_labels.json not found: {LABELS_JSON}")
        return False

    data  = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    items = data.get("items", [])

    entry = find_entry(items, batch_id, shot_arg)
    if entry is None:
        print(f"  No entry found for batch={batch_id}  shot={shot_arg}")
        return False

    for desc, p in collect_paths(entry):
        if not p.exists():
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"  Deleted {desc}: {p.name}")

    rewrite_labels(items, entry["rel_shot_dir"])
    print(f"  Removed from dataset. {len(items) - 1} entries remaining.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Remove a clip from the dataset.")
    ap.add_argument("--batch", required=True, help="Batch ID (e.g. live_20260321_181637)")
    ap.add_argument("--shot",  required=True, help="Dataset name or shot number (e.g. shot_000004 or 018)")
    args = ap.parse_args()

    if not LABELS_JSON.exists():
        print(f"shot_labels.json not found: {LABELS_JSON}")
        sys.exit(1)

    data  = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    items = data.get("items", [])

    entry = find_entry(items, args.batch, args.shot)
    if entry is None:
        print(f"No entry found for batch={args.batch}  shot={args.shot}")
        sys.exit(1)

    print(f"Found:  {entry['dataset_name']}  ({entry['label']})")
    print(f"  rel_shot_dir: {entry['rel_shot_dir']}\n")

    paths = collect_paths(entry)
    print("Will delete:")
    for desc, p in paths:
        exists = "exists" if p.exists() else "not found"
        print(f"  [{exists}]  {desc}: {p}")

    print()
    try:
        confirm = input("Confirm deletion? [y/N]: ").strip().lower()
    except EOFError:
        confirm = ""

    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    for desc, p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"  Deleted {desc}: {p}")

    rewrite_labels(items, entry["rel_shot_dir"])
    print(f"\nRemoved from shot_labels.json/csv. {len(items) - 1} entries remaining.")


if __name__ == "__main__":
    main()
