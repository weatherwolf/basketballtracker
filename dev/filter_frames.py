# filter_frames.py
import argparse
import os
import re
import shutil
import json
from pathlib import Path

import cv2
import numpy as np


# --- folders (edit here) ---
REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = REPO_ROOT / "work"
# Defaults (can be overridden via CLI args)
DEFAULT_FRAMES_RAW_DIR = WORK_DIR / "frames_raw"
# Output datasets live here (per-shot subfolders)
DEFAULT_FRAMES_BATCH_DIR = WORK_DIR / "frames_batch"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Filter raw frames and split them into per-shot folders.")
    ap.add_argument(
        "--frames-raw-dir",
        default=str(DEFAULT_FRAMES_RAW_DIR),
        help=f"Directory containing extracted frames (default: {DEFAULT_FRAMES_RAW_DIR})",
    )
    ap.add_argument(
        "--frames-batch-dir",
        default=str(DEFAULT_FRAMES_BATCH_DIR),
        help=f"Output directory for per-shot subfolders (default: {DEFAULT_FRAMES_BATCH_DIR})",
    )
    ap.add_argument(
        "--clear-output",
        action="store_true",
        help="If set, clears the output frames-batch-dir before writing new shot folders.",
    )
    return ap.parse_args()

# --- ball color threshold ---
LOWER_ORANGE = np.array([5, 120, 120])
UPPER_ORANGE = np.array([20, 255, 255])

# --- detection knobs ---
MIN_CONTOUR_AREA = 150
MIN_MASK_PIXELS = 200

# --- multi-shot splitting ---
# If there are MORE than this many consecutive frames without a ball BETWEEN two ball frames,
# treat them as separate shots.
NO_BALL_GAP_THRESHOLD = 100

# --- filename format ---
FRAME_PAD = 6   # matches ffmpeg %06d


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _has_ball(img_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    if int(cv2.countNonZero(mask)) < MIN_MASK_PIXELS:
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    biggest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(biggest) >= MIN_CONTOUR_AREA


def _parse_frame_filename(filename: str):
    """
    Parses filenames like:
      frame_<VIDEO_NAME>_000123.jpg
    Returns (prefix, frame_index) where prefix is "frame_<VIDEO_NAME>".
    """
    stem = Path(filename).stem
    m = re.match(r"^(.*)_(\d+)$", stem)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _make_dataset_frame_name(dataset_name: str, new_index: int, ext: str) -> str:
    """
    Create a new frame name for a specific dataset (shot), indexed from 0.
    Example:
      dataset_name="frame_myvideo_shot000", new_index=0
        -> "frame_myvideo_shot000_000000.jpg"
    """
    return f"{dataset_name}_{new_index:0{FRAME_PAD}d}{ext}"


def _safe_dirname(name: str) -> str:
    # Windows-safe-ish folder name (keep it simple)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "dataset"


def _split_selected_into_shots(selected_items, gap_threshold: int):
    """
    selected_items: list[(frame_idx:int, path:Path)] sorted by frame_idx
    Returns list of shots, each shot is list[(frame_idx, path)]
    """
    shots = []
    current = []
    prev_ball_idx = None

    for frame_idx, p in selected_items:
        if prev_ball_idx is not None:
            gap = frame_idx - prev_ball_idx - 1  # frames between ball frames
            if gap > gap_threshold and current:
                shots.append(current)
                current = []

        current.append((frame_idx, p))
        prev_ball_idx = frame_idx

    if current:
        shots.append(current)

    return shots


def main() -> None:
    args = _parse_args()
    ALL_FRAMES_DIR = Path(args.frames_raw_dir).resolve()
    FRAMES_DIR = Path(args.frames_batch_dir).resolve()

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Optional overwrite semantics (useful if you target a shared output dir).
    if args.clear_output:
        for p in list(FRAMES_DIR.iterdir()):
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.is_file():
                    p.unlink()
            except Exception:
                # best-effort cleanup; continue processing
                pass

    if not ALL_FRAMES_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {ALL_FRAMES_DIR}")

    all_files = [
        p for p in ALL_FRAMES_DIR.iterdir()
        if p.is_file() and _is_image_file(p)
    ]
    all_files.sort(key=lambda p: p.name)

    if not all_files:
        print(f"No images found in: {ALL_FRAMES_DIR}")
        return

    # Group by prefix (so multiple videos in all_frames are handled safely)
    by_prefix = {}
    rejected = []

    for p in all_files:
        prefix, frame_idx = _parse_frame_filename(p.name)
        if prefix is None or frame_idx is None:
            rejected.append(p)
            continue

        img = cv2.imread(str(p))
        if img is None:
            rejected.append(p)
            continue

        if _has_ball(img):
            by_prefix.setdefault(prefix, []).append((frame_idx, p))
        else:
            rejected.append(p)

    moved = 0
    manifest = {
        "repo_root": str(REPO_ROOT),
        "work_dir": str(WORK_DIR),
        "all_frames_dir": str(ALL_FRAMES_DIR),
        "frames_dir": str(FRAMES_DIR),
        "gap_threshold": NO_BALL_GAP_THRESHOLD,
        "datasets": [],
    }
    total_shots = 0

    # For each video prefix, split into shot datasets and move frames into subfolders
    for prefix in sorted(by_prefix.keys()):
        items = by_prefix[prefix]
        items.sort(key=lambda t: t[0])
        shots = _split_selected_into_shots(items, NO_BALL_GAP_THRESHOLD)

        for shot_idx, shot_items in enumerate(shots):
            dataset_name = f"{prefix}_shot{shot_idx:03d}"
            dataset_dir = FRAMES_DIR / _safe_dirname(dataset_name)
            dataset_dir.mkdir(parents=True, exist_ok=True)

            manifest["datasets"].append(
                {
                    "dataset_name": dataset_name,
                    "dataset_dir": str(dataset_dir),
                    "source_prefix": prefix,
                    "shot_index": shot_idx,
                    "num_frames": len(shot_items),
                    "source_first_frame": shot_items[0][0] if shot_items else None,
                    "source_last_frame": shot_items[-1][0] if shot_items else None,
                }
            )
            total_shots += 1

            # Move frames for this shot, renumbered from 0
            for new_idx, (_src_frame_idx, src_path) in enumerate(shot_items):
                new_name = _make_dataset_frame_name(dataset_name, new_idx, src_path.suffix)
                dst = dataset_dir / new_name

                # overwrite existing destination file (Windows-safe)
                if dst.exists():
                    dst.unlink()

                shutil.move(str(src_path), str(dst))
                moved += 1

    # Clear all remaining files
    for p in ALL_FRAMES_DIR.iterdir():
        if p.is_file():
            p.unlink()

    # Write a small manifest to help with later labeling / debugging
    manifest_path = FRAMES_DIR / "shots_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Scanned:  {len(all_files)} frames")
    print(f"Kept:    {moved} frames -> {FRAMES_DIR} (split into {total_shots} shot datasets)")
    print(f"Cleared: {ALL_FRAMES_DIR} (now empty)")
    print(f"Wrote:   {manifest_path}")


if __name__ == "__main__":
    main()
