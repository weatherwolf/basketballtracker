"""
Review wrong MiniRocket predictions by playing each shot's preview clip.

Reads data/minirocket_wrong_predictions.csv, looks up the preview.mp4 for each
shot in data/shot_labels.csv, then opens them one by one.

Run from repo root:
    python dev/review_wrong_predictions.py
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT             = Path(__file__).resolve().parent.parent
WRONG_PREDS_PATH      = REPO_ROOT / "data" / "minirocket_wrong_predictions.csv"
WRONG_PREDS_FINAL_PATH = REPO_ROOT / "data" / "minirocket_wrong_predictions_final_test.csv"
LABELS_PATH           = REPO_ROOT / "data" / "shot_labels.csv"


def load_wrong_predictions():
    rows = []
    with open(WRONG_PREDS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def build_preview_index():
    """Map (batch_id, shot_number) -> rel_preview_mp4 from shot_labels.csv."""
    index = {}
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset_name = row["dataset_name"]
            rel_shot_dir = row["rel_shot_dir"]
            parts        = Path(rel_shot_dir).parts
            batch_id     = next((p for p in parts if p.startswith(("pending_", "live_"))), None)
            if not batch_id:
                continue
            shot_number = dataset_name.split("_shot")[-1] if "_shot" in dataset_name else None
            if shot_number is None:
                continue
            index[(batch_id, shot_number)] = row["rel_preview_mp4"]
    return index


def open_video(path: Path) -> None:
    os.startfile(str(path))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true", help="Review final held-out test wrong predictions instead of CV")
    args = ap.parse_args()

    global WRONG_PREDS_PATH
    if args.final_test:
        WRONG_PREDS_PATH = WRONG_PREDS_FINAL_PATH

    wrong = load_wrong_predictions()
    if not wrong:
        print("No wrong predictions found in", WRONG_PREDS_PATH)
        return

    wrong.sort(key=lambda r: float(r.get("confidence", 0)), reverse=True)

    index = build_preview_index()

    print(f"Wrong predictions: {len(wrong)}\n")

    not_found = []
    for i, pred in enumerate(wrong, start=1):
        batch_id    = pred["batch_id"]
        shot_number = pred["shot_number"]
        true_label  = pred["true_label"]
        key         = (batch_id, shot_number)

        rel_preview = index.get(key)
        if not rel_preview:
            not_found.append(f"  {batch_id}  shot {shot_number}")
            continue

        preview_path = REPO_ROOT / rel_preview
        if not preview_path.exists():
            not_found.append(f"  {batch_id}  shot {shot_number}  (file missing: {rel_preview})")
            continue

        confidence     = pred.get("confidence", "?")
        decision_score = pred.get("decision_score", "?")
        print(f"[{i}/{len(wrong)}]  batch={batch_id}  shot={shot_number}  true_label={true_label}  confidence={confidence}  decision_score={decision_score}")
        open_video(preview_path)

        try:
            raw = input("  Press Enter for next, q to quit: ").strip().lower()
        except EOFError:
            break
        if raw == "q":
            break

    if not_found:
        print(f"\nCould not find preview for {len(not_found)} shots:")
        for s in not_found:
            print(s)


if __name__ == "__main__":
    main()
