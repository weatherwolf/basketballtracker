"""
Verify that shots where the minimum ball-to-hoop distance across all frames
is >= --dist are always misses.

Prints any violations found and a final pass/fail summary.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT    = Path(__file__).resolve().parent.parent
CSV_PATH     = REPO_ROOT / "data" / "shot_labels.csv"

LOWER_ORANGE = np.array([5, 120, 120])
UPPER_ORANGE = np.array([20, 255, 255])
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}


def load_ellipse(ellipse_meta_rel: str):
    path = REPO_ROOT / ellipse_meta_rel
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    e = data["ellipse"]
    return float(e["center"][0]), float(e["center"][1])


def min_ball_distance(shot_dir: Path, hoop_cx: float, hoop_cy: float):
    frames = sorted(
        (p for p in shot_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )
    min_dist = float("inf")
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
        (x, y), _ = cv2.minEnclosingCircle(cnt)
        dist = math.hypot(x - hoop_cx, y - hoop_cy)
        if dist < min_dist:
            min_dist = dist
    return min_dist if min_dist < float("inf") else None


def main():
    ap = argparse.ArgumentParser(
        description="Verify that shots with min ball-to-hoop distance >= DIST are always misses."
    )
    ap.add_argument("--dist", type=float, default=35.0,
                    help="Distance threshold in pixels (default: 35)")
    args = ap.parse_args()
    threshold = args.dist

    shots = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "skip" or not row["ellipse_meta"]:
                continue
            shots.append(row)

    print(f"Checking {len(shots)} shots (threshold = {threshold}px)\n")

    violations = []
    skipped    = []
    furthest_make = 0

    for shot in shots:
        dataset_name = shot["dataset_name"]
        label        = shot["label"]
        shot_dir     = REPO_ROOT / shot["rel_shot_dir"]

        ellipse = load_ellipse(shot["ellipse_meta"])
        if ellipse is None:
            skipped.append(f"{dataset_name}: ellipse file missing")
            continue

        if not shot_dir.exists():
            skipped.append(f"{dataset_name}: frames folder missing")
            continue

        hoop_cx, hoop_cy = ellipse
        min_dist = min_ball_distance(shot_dir, hoop_cx, hoop_cy)

        if min_dist is None:
            skipped.append(f"{dataset_name}: no ball detected in any frame")
            continue

        rule_says_miss = min_dist >= threshold
        if rule_says_miss and label != "miss":
            violations.append(
                f"  VIOLATION  {dataset_name}  label={label.upper()}  min_dist={min_dist:.1f}px"
            )
        else:
            status = f"{'MISS (rule)' if rule_says_miss else 'close shot  ':11s}"
            print(f"  ok  {status}  {dataset_name:40s}  min_dist={min_dist:.1f}px  label={label}")
        if label == "goal":
            if min_dist > furthest_make:
                furthest_make = min_dist

    print()
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")
        print()

    if violations:
        print(f"RULE VIOLATIONS ({len(violations)}) — rule does NOT always hold:")
        for v in violations:
            print(v)
        sys.exit(1)
    else:
        print(f"PASSED — rule holds across all {len(shots) - len(skipped)} checked shots.")
        print(f"Furthest make: {furthest_make:.1f}px")

if __name__ == "__main__":
    main()
