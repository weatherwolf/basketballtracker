"""
test_sticker_hsv.py

Quick visual test for sticker HSV detection on a batch.

For each shot folder in --frames-dir it processes ALL frames, applies an HSV
mask tuned for the sticker colour (#5f8556 ~= H=54 S=90 V=133 in OpenCV scale),
and writes side-by-side images (original | mask overlay) into per-shot subfolders
under --out-dir.

Adjust --h-center, --h-range, --s-lo/hi, --v-lo/hi to tune the mask.

Usage:
    python dev/test_sticker_hsv.py
    python dev/test_sticker_hsv.py --h-center 54 --h-range 12 --s-lo 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _all_frames(shot_dir: Path) -> List[Path]:
    return sorted(
        [p for p in shot_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )


def make_overlay(img_bgr: np.ndarray, mask: np.ndarray, color_bgr=(0, 255, 0), alpha=0.5) -> np.ndarray:
    overlay = img_bgr.copy()
    overlay[mask > 0] = color_bgr
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


def annotate(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def process_shot(shot_dir: Path, out_dir: Path, lo: np.ndarray, hi: np.ndarray, label: str) -> None:
    frames = _all_frames(shot_dir)
    if not frames:
        print(f"  Skipping {shot_dir.name}: no frames found")
        return

    shot_out = out_dir / shot_dir.name
    shot_out.mkdir(parents=True, exist_ok=True)

    pcts = []
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lo, hi)

        pct = 100.0 * np.count_nonzero(mask) / mask.size
        pcts.append(pct)

        left = annotate(img, f"{frame_path.stem}")
        right = annotate(make_overlay(img, mask, color_bgr=(255, 0, 255), alpha=0.8), f"{label}  hit={pct:.2f}%")

        side_by_side = np.hstack([left, right])
        out_path = shot_out / frame_path.name
        cv2.imwrite(str(out_path), side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 90])

    if pcts:
        print(f"  {shot_dir.name}: {len(frames)} frames | hit avg={sum(pcts)/len(pcts):.2f}% max={max(pcts):.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Test sticker HSV detection on all frames of a batch.")
    ap.add_argument("--frames-dir", default="work/runs/pending_115665147/frames_batch",
                    help="Shot folders to test (default: pending_115665147)")
    ap.add_argument("--out-dir", default="work/sticker_hsv_test",
                    help="Output folder for annotated images (default: work/sticker_hsv_test)")
    # Sticker colour: #5f8556 -> OpenCV HSV ~= H=54 S=90 V=133
    ap.add_argument("--h-center", type=int, default=54, help="HSV hue centre (OpenCV 0-179, default 54)")
    ap.add_argument("--h-range",  type=int, default=15, help="+-hue range around centre (default 15)")
    ap.add_argument("--s-lo",     type=int, default=55, help="Min saturation 0-255 (default 55)")
    ap.add_argument("--s-hi",     type=int, default=175, help="Max saturation 0-255 (default 175)")
    ap.add_argument("--v-lo",     type=int, default=60, help="Min value 0-255 (default 60)")
    ap.add_argument("--v-hi",     type=int, default=190, help="Max value 0-255 (default 190)")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)

    if not frames_dir.exists():
        print(f"frames-dir not found: {frames_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    h_lo = max(0, args.h_center - args.h_range)
    h_hi = min(179, args.h_center + args.h_range)
    lo = np.array([h_lo, args.s_lo, args.v_lo], dtype=np.uint8)
    hi = np.array([h_hi, args.s_hi, args.v_hi], dtype=np.uint8)
    label = f"H[{h_lo}-{h_hi}] S[{args.s_lo}-{args.s_hi}] V[{args.v_lo}-{args.v_hi}]"

    print(f"HSV range: {label}")
    print(f"Source:    {frames_dir}")
    print(f"Output:    {out_dir}")
    print()

    shot_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not shot_dirs:
        print("No shot folders found.")
        return 1

    total_frames = 0
    for shot_dir in shot_dirs:
        process_shot(shot_dir, out_dir, lo, hi, label)
        total_frames += len(_all_frames(shot_dir))

    print(f"\nDone: {len(shot_dirs)} shots, {total_frames} frames -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
