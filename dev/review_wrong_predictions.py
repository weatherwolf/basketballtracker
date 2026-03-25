"""
Review wrong MiniRocket predictions by playing each shot's preview clip.

Reads data/minirocket_wrong_predictions.csv, looks up the preview.mp4 for each
shot in data/shot_labels.csv, then opens them one by one.

Run from repo root:
    python dev/review_wrong_predictions.py
"""

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
from delete_clip import delete_clip
from reverse_label import flip_label
from show_closest_frames import load_ellipse, compute_centers, annotate_frame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REPO_ROOT, LABELS_CSV as LABELS_PATH

WRONG_PREDS_PATH       = REPO_ROOT / "data" / "minirocket_wrong_predictions.csv"
WRONG_PREDS_FINAL_PATH = REPO_ROOT / "data" / "minirocket_wrong_predictions_final_test.csv"


def load_wrong_predictions():
    rows = []
    with open(WRONG_PREDS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def build_preview_index():
    """Map (batch_id, shot_number) -> dict with video path, rel_shot_dir, ellipse_meta.
    Prefers rel_preview_mp4; falls back to rel_export_mp4 (used for live_ batches
    which have no preview.mp4 but do have a labeled export mp4).
    """
    index = {}
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset_name = row["dataset_name"]
            rel_shot_dir = row["rel_shot_dir"]
            parts        = Path(rel_shot_dir).parts
            batch_id     = next((p for p in parts if p.startswith(("pending_", "live_"))), None)
            if not batch_id:
                continue
            shot_number = dataset_name.split("_shot")[-1] if "_shot" in dataset_name else dataset_name
            video = row["rel_preview_mp4"] or row["rel_export_mp4"]
            index[(batch_id, shot_number)] = {
                "video":        video,
                "rel_shot_dir": rel_shot_dir,
                "ellipse_meta": row["ellipse_meta"],
            }
    return index


def show_closest_frame(rel_shot_dir: str, ellipse_meta: str, header: str) -> None:
    ellipse = load_ellipse(ellipse_meta)
    if ellipse is None:
        print("  (no ellipse found, skipping frame display)")
        return
    shot_dir = REPO_ROOT / rel_shot_dir
    if not shot_dir.exists():
        print(f"  (frames folder missing: {shot_dir})")
        return
    centers = [c for c in compute_centers(shot_dir) if c["radius"] >= 20]
    if not centers:
        print("  (no ball detected in any frame)")
        return
    cx, cy = ellipse[0]
    for row in centers:
        row["dist"] = math.hypot(row["x"] - cx, row["y"] - cy)
    best = min(centers, key=lambda r: r["dist"])
    img = cv2.imread(str(best["path"]))
    if img is None:
        return
    ann = annotate_frame(img, ellipse, best["x"], best["y"], best["radius"], best["dist"])
    diameter_px = best["radius"] * 2
    cv2.putText(ann, f"diam={diameter_px:.1f}px", (int(best["x"]) + 8, int(best["y"]) + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(ann, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Closest frame to hoop", ann)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def open_video(path: Path) -> None:
    os.startfile(str(path))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true", help="Review final held-out test wrong predictions instead of CV")
    ap.add_argument("--show-ellipse", action="store_true", help="Show the closest ball-to-hoop frame with ellipse overlay instead of opening the video")
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

        entry = index.get(key)
        if not entry:
            not_found.append(f"  {batch_id}  shot {shot_number}")
            continue

        preview_path = REPO_ROOT / entry["video"] if entry["video"] else None
        if not args.show_ellipse and (not preview_path or not preview_path.exists()):
            not_found.append(f"  {batch_id}  shot {shot_number}  (file missing: {entry['video']})")
            continue

        confidence     = pred.get("confidence", "?")
        decision_score = pred.get("decision_score", "?")
        print(f"[{i}/{len(wrong)}]  batch={batch_id}  shot={shot_number}  true_label={true_label}  confidence={confidence}  decision_score={decision_score}")

        if args.show_ellipse:
            header = f"{batch_id}  |  shot {shot_number}  [{true_label.upper()}]"
            show_closest_frame(entry["rel_shot_dir"], entry["ellipse_meta"], header)
        else:
            open_video(preview_path)

        try:
            raw = input("  Press Enter for next, r to reverse label, d to delete, q to quit: ").strip().lower()
        except EOFError:
            break
        if raw == "q":
            break
        if raw == "r":
            flip_label(batch_id, shot_number)
        if raw == "d":
            try:
                confirm = input(f"  Confirm delete {batch_id} / {shot_number}? [y/N]: ").strip().lower()
            except EOFError:
                confirm = ""
            if confirm == "y":
                delete_clip(batch_id, shot_number)
            else:
                print("  Aborted.")

    if not_found:
        print(f"\nCould not find preview for {len(not_found)} shots:")
        for s in not_found:
            print(s)


if __name__ == "__main__":
    main()
