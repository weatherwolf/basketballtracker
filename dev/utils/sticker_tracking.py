"""
sticker_tracking.py

Per-frame sticker visibility tracking for a shot.

For each frame, detects sticker blobs via HSV and matches them to a reference
set of 8 positions established from the first frame. Reference positions are
sorted clockwise starting from the topmost blob (smallest y in image coords).
Each sticker is recorded as found (True) or occluded (False) based on whether
a blob is detected within `threshold` pixels of its reference position.

Output CSV: data/sticker_tracking/<batch_id>_<dataset_name>.csv
Columns:    frame_index, sticker_1, sticker_2, ..., sticker_8

Importable API:
    build_reference(frame, lo, hi)                       -> list of 8 (x, y)
    track_frame(frame, lo, hi, reference, threshold)     -> list of 8 bools
    track_shot(shot_dir, lo, hi, threshold)              -> list of dicts

CLI usage:
    python dev/utils/sticker_tracking.py
    python dev/utils/sticker_tracking.py --batch live_20260322_210917
    python dev/utils/sticker_tracking.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import math
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT, LABELS_CSV, LABELS_JSON, LABEL_CSV_FIELDS, STICKER_TRACKING_DIR, STICKER_MORPH_CLOSE_PX

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# HSV defaults (same as test_sticker_hsv.py and fit_ellipse.py)
DEFAULT_H_CENTER = 54
DEFAULT_H_RANGE  = 15
DEFAULT_S_LO     = 55
DEFAULT_S_HI     = 175
DEFAULT_V_LO     = 60
DEFAULT_V_HI     = 190
DEFAULT_THRESHOLD = 15.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_hsv_bounds(h_center=DEFAULT_H_CENTER, h_range=DEFAULT_H_RANGE,
                     s_lo=DEFAULT_S_LO, s_hi=DEFAULT_S_HI,
                     v_lo=DEFAULT_V_LO, v_hi=DEFAULT_V_HI):
    lo = np.array([max(0,   h_center - h_range), s_lo, v_lo], dtype=np.uint8)
    hi = np.array([min(179, h_center + h_range), s_hi, v_hi], dtype=np.uint8)
    return lo, hi


def _detect_blob_centers(frame: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[tuple[float, float]]:
    """Return (x, y) centroid of each sticker blob passing the area filter."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (STICKER_MORPH_CLOSE_PX, STICKER_MORPH_CLOSE_PX))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_kernel,  iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) < 20:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    return centers


def _sort_clockwise_from_top(centers: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort centers clockwise starting from the topmost blob (smallest y)."""
    cx = sum(p[0] for p in centers) / len(centers)
    cy = sum(p[1] for p in centers) / len(centers)

    def angle(p):
        return math.atan2(p[1] - cy, p[0] - cx)

    top_angle = angle(min(centers, key=lambda p: p[1]))

    return sorted(centers, key=lambda p: (angle(p) - top_angle) % (2 * math.pi))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_reference(frame: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[tuple[float, float]]:
    """Detect sticker blobs and return 8 (x, y) sorted clockwise from topmost.

    Raises ValueError if the blob count is not exactly 8.
    """
    centers = _detect_blob_centers(frame, lo, hi)
    if len(centers) != 8:
        raise ValueError(f"Expected 8 sticker blobs in reference frame, found {len(centers)}")
    return _sort_clockwise_from_top(centers)


def track_frame(frame: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                reference: list[tuple[float, float]],
                threshold: float = DEFAULT_THRESHOLD) -> list[bool]:
    """Return 8 bools: True if sticker i was detected within threshold px of reference[i].

    Each detected blob is claimed by at most one reference (greedy nearest-neighbour).
    """
    centers = _detect_blob_centers(frame, lo, hi)
    found   = [False] * 8
    claimed: set[int] = set()

    for i, (rx, ry) in enumerate(reference):
        best_dist = float("inf")
        best_j    = -1
        for j, (bx, by) in enumerate(centers):
            if j in claimed:
                continue
            d = math.sqrt((bx - rx) ** 2 + (by - ry) ** 2)
            if d < best_dist:
                best_dist = d
                best_j    = j
        if best_j >= 0 and best_dist < threshold:
            found[i] = True
            claimed.add(best_j)

    return found


def track_shot(shot_dir: Path, lo: np.ndarray, hi: np.ndarray,
               threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Process all frames in shot_dir and return per-frame sticker visibility.

    Reference positions are established from the first readable frame.
    Returns [] if the first frame does not contain exactly 8 stickers.

    Each returned dict has keys: frame_index, sticker_1 .. sticker_8 (bool).
    """
    frames = sorted(
        (p for p in shot_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )
    if not frames:
        return []

    reference = None
    rows: list[dict] = []

    for path in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue

        stem  = path.stem
        parts = stem.rsplit("_", 1)
        frame_index = int(parts[-1]) if parts[-1].isdigit() else len(rows)

        if reference is None:
            try:
                reference = build_reference(img, lo, hi)
            except ValueError as e:
                print(f"  WARNING: {e} in first frame {path.name} — skipping shot")
                return []

        visible = track_frame(img, lo, hi, reference, threshold)
        row = {"frame_index": frame_index}
        for i, v in enumerate(visible, 1):
            row[f"sticker_{i}"] = v
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Backfill helper
# ---------------------------------------------------------------------------

def _first_frame(shot_dir: Path) -> Path | None:
    frames = sorted(
        (p for p in shot_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )
    return frames[0] if frames else None


def backfill_8_stickers(lo: np.ndarray, hi: np.ndarray) -> None:
    """One-time pass: detect blob count in first frame of each shot and write
    has_8_stickers to shot_labels.csv and shot_labels.json.

    Shots with has_stickers=False get has_8_stickers=False without frame inspection.
    Errors if a shot's frames folder is missing.
    """
    # Read CSV into ordered list of dicts
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Read JSON for write-back
    json_data = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    json_index = {it["rel_shot_dir"]: it for it in json_data.get("items", [])}

    updated = 0
    for row in rows:
        has_stickers = row.get("has_stickers", "").lower() in ("true", "1", "yes")
        if not has_stickers:
            row["has_8_stickers"] = False
            if row["rel_shot_dir"] in json_index:
                json_index[row["rel_shot_dir"]]["has_8_stickers"] = False
            continue

        shot_dir = REPO_ROOT / row["rel_shot_dir"]
        if not shot_dir.exists():
            print(f"  ERROR: frames folder missing for {row['dataset_name']} — cannot determine has_8_stickers")
            continue

        first = _first_frame(shot_dir)
        if first is None:
            print(f"  ERROR: no frames found in {shot_dir} for {row['dataset_name']}")
            continue

        img = cv2.imread(str(first))
        if img is None:
            print(f"  ERROR: could not read {first} for {row['dataset_name']}")
            continue

        count = len(_detect_blob_centers(img, lo, hi))
        has_8 = count == 8
        row["has_8_stickers"] = has_8
        if row["rel_shot_dir"] in json_index:
            json_index[row["rel_shot_dir"]]["has_8_stickers"] = has_8
        print(f"  {row['dataset_name']}: {count} stickers → has_8_stickers={has_8}")
        updated += 1

    # Write CSV
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Write JSON
    LABELS_JSON.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    print(f"\nDone. Inspected {updated} shots with stickers.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIELDNAMES = ["frame_index"] + [f"sticker_{i}" for i in range(1, 9)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract per-frame sticker visibility into data/sticker_tracking/."
    )
    ap.add_argument("--backfill-8-stickers", action="store_true",
                    help="One-time pass: detect blob count per shot and write has_8_stickers to shot_labels.csv/json")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract even if a CSV already exists for a shot")
    ap.add_argument("--batch", metavar="BATCH_ID",
                    help="Limit processing to a specific batch (e.g. live_20260322_210917)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Max px distance to match blob to reference position (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--h-center", type=int, default=DEFAULT_H_CENTER)
    ap.add_argument("--h-range",  type=int, default=DEFAULT_H_RANGE)
    ap.add_argument("--s-lo",     type=int, default=DEFAULT_S_LO)
    ap.add_argument("--s-hi",     type=int, default=DEFAULT_S_HI)
    ap.add_argument("--v-lo",     type=int, default=DEFAULT_V_LO)
    ap.add_argument("--v-hi",     type=int, default=DEFAULT_V_HI)
    args = ap.parse_args()

    lo, hi = _make_hsv_bounds(args.h_center, args.h_range,
                              args.s_lo, args.s_hi, args.v_lo, args.v_hi)

    if args.backfill_8_stickers:
        backfill_8_stickers(lo, hi)
        return

    STICKER_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    shots = []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "skip":
                continue
            if row.get("has_8_stickers", "").lower() not in ("true", "1", "yes"):
                continue
            shots.append(row)

    print(f"Processing {len(shots)} shots with 8 stickers\n")

    written = 0
    skipped = 0
    failed  = 0

    for shot in shots:
        dataset_name = shot["dataset_name"]
        shot_dir     = REPO_ROOT / shot["rel_shot_dir"]
        parts        = Path(shot["rel_shot_dir"]).parts
        batch_id     = next((p for p in parts if p.startswith(("pending_", "live_"))), "unknown")
        out_path     = STICKER_TRACKING_DIR / f"{batch_id}_{dataset_name}.csv"

        if args.batch and batch_id != args.batch:
            continue

        if out_path.exists() and not args.overwrite:
            print(f"  skip (exists)         {dataset_name}")
            skipped += 1
            continue

        if not shot_dir.exists():
            print(f"  skip (no frames folder) {dataset_name}")
            skipped += 1
            continue

        rows = track_shot(shot_dir, lo, hi, args.threshold)

        if not rows:
            print(f"  skip (reference failed) {dataset_name}")
            failed += 1
            continue

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  wrote {len(rows):4d} frames  {dataset_name}")
        written += 1

    print(f"\nDone. written={written}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
