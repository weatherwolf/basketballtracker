"""
Extract ball center (x, y, radius) per frame for every labeled shot and save
to data/ball_tracking/<batch_id>_<dataset_name>.csv.

Uses the same HSV ball detection as show_closest_frames.py.
Skips shots that are already tracked unless --overwrite is passed.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT     = Path(__file__).resolve().parent.parent
CSV_PATH      = REPO_ROOT / "data" / "shot_labels.csv"
TRACKING_DIR  = REPO_ROOT / "data" / "ball_tracking"

LOWER_ORANGE  = np.array([5, 120, 120])
UPPER_ORANGE  = np.array([20, 255, 255])
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}


def compute_centers(shot_dir: Path):
    frames = sorted(
        (p for p in shot_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )
    rows = []
    for path in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        stem  = path.stem
        parts = stem.rsplit("_", 1)
        frame_index = int(parts[-1]) if parts[-1].isdigit() else len(rows)
        rows.append({
            "frame_index": frame_index,
            "x":           round(float(x), 2),
            "y":           round(float(y), 2),
            "radius":      round(float(radius), 2),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Extract per-frame ball tracking data into data/ball_tracking/."
    )
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract even if a CSV already exists for a shot")
    args = ap.parse_args()

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    shots = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "skip":
                continue
            shots.append(row)

    print(f"Processing {len(shots)} shots\n")

    written  = 0
    skipped  = 0
    no_ball  = 0

    for shot in shots:
        dataset_name = shot["dataset_name"]
        shot_dir     = REPO_ROOT / shot["rel_shot_dir"]
        parts        = Path(shot["rel_shot_dir"]).parts
        batch_id     = next((p for p in parts if p.startswith("pending_")), "unknown")
        out_path     = TRACKING_DIR / f"{batch_id}_{dataset_name}.csv"

        if out_path.exists() and not args.overwrite:
            print(f"  skip (exists)   {dataset_name}")
            skipped += 1
            continue

        if not shot_dir.exists():
            print(f"  skip (no frames folder)  {dataset_name}")
            skipped += 1
            continue

        centers = compute_centers(shot_dir)

        if not centers:
            print(f"  skip (no ball detected)  {dataset_name}")
            no_ball += 1
            continue

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_index", "x", "y", "radius"])
            writer.writeheader()
            writer.writerows(centers)

        print(f"  wrote {len(centers):4d} frames  {dataset_name}")
        written += 1

    print(f"\nDone. written={written}  skipped={skipped}  no_ball={no_ball}")


if __name__ == "__main__":
    main()
