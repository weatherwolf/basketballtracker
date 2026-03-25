"""
debug_sticker_count.py

Shows any shot whose first frame contains an unexpected sticker count.
Displays a side-by-side image (original with blob centres + HSV mask)
and prints the shot name so you can investigate.

Usage:
    python dev/utils/debug_sticker_count.py           # default: flag count 9
    python dev/utils/debug_sticker_count.py --count 9
    python dev/utils/debug_sticker_count.py --count 6
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT, LABELS_CSV
from sticker_tracking import _detect_blob_centers, _make_hsv_bounds, IMAGE_EXTS


def first_frame(shot_dir: Path) -> Path | None:
    frames = sorted(
        (p for p in shot_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )
    return frames[0] if frames else None


def show_debug(img: np.ndarray, centers: list, dataset_name: str, lo: np.ndarray, hi: np.ndarray) -> None:
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)

    annotated = img.copy()
    for (x, y) in centers:
        cv2.circle(annotated, (int(x), int(y)), 8, (255, 0, 255), -1)
        cv2.circle(annotated, (int(x), int(y)), 9, (0, 0, 0), 1)
    cv2.putText(annotated, f"{dataset_name}  —  {len(centers)} stickers  |  Press any key",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    side_by_side = np.hstack([annotated, mask_bgr])
    cv2.imshow("Sticker Count Debug", side_by_side)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=9,
                    help="Flag shots where sticker count equals this value (default: 9)")
    ap.add_argument("--h-center", type=int, default=54)
    ap.add_argument("--h-range",  type=int, default=15)
    ap.add_argument("--s-lo",     type=int, default=55)
    ap.add_argument("--s-hi",     type=int, default=175)
    ap.add_argument("--v-lo",     type=int, default=60)
    ap.add_argument("--v-hi",     type=int, default=190)
    args = ap.parse_args()

    lo, hi = _make_hsv_bounds(args.h_center, args.h_range,
                              args.s_lo, args.s_hi, args.v_lo, args.v_hi)

    shots = []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("has_stickers", "").lower() not in ("true", "1", "yes"):
                continue
            shots.append(row)

    print(f"Checking {len(shots)} shots with stickers for count == {args.count}\n")

    flagged = 0
    for shot in shots:
        shot_dir = REPO_ROOT / shot["rel_shot_dir"]
        if not shot_dir.exists():
            print(f"  ERROR: frames folder missing for {shot['dataset_name']}")
            continue

        frame_path = first_frame(shot_dir)
        if frame_path is None:
            print(f"  ERROR: no frames in {shot_dir}")
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  ERROR: could not read {frame_path}")
            continue

        centers = _detect_blob_centers(img, lo, hi)
        if len(centers) == args.count:
            print(f"  FLAGGED {shot['dataset_name']}: {len(centers)} stickers")
            show_debug(img, centers, shot["dataset_name"], lo, hi)
            flagged += 1

    print(f"\nDone. {flagged} shot(s) flagged with count == {args.count}.")


if __name__ == "__main__":
    main()
