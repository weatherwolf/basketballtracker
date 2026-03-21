"""
Extract ball center (x, y, radius) per frame for every labeled shot and save
to data/ball_tracking/<batch_id>_<dataset_name>.csv.

Also writes hoop-normalized coordinates (xn, yn, dist_n) to
data/ball_tracking_normalized/<batch_id>_<dataset_name>.csv when an ellipse
is available for the shot.

Uses the same HSV ball detection as show_closest_frames.py.
Skips shots that are already tracked unless --overwrite is passed.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT          = Path(__file__).resolve().parent.parent
CSV_PATH           = REPO_ROOT / "data" / "shot_labels.csv"
TRACKING_DIR       = REPO_ROOT / "data" / "ball_tracking"
NORMALIZED_DIR     = REPO_ROOT / "data" / "ball_tracking_normalized"

LOWER_ORANGE  = np.array([5, 120, 120])
UPPER_ORANGE  = np.array([20, 255, 255])
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}


def load_ellipse(ellipse_meta: str):
    """Load ellipse from a relative path. Returns (cx, cy, ax0, ax1, angle) or None."""
    if not ellipse_meta:
        return None
    path = REPO_ROOT / ellipse_meta
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        e = data["ellipse"]
        cx, cy = e["center"]
        ax0, ax1 = e["axes"]
        return float(cx), float(cy), float(ax0), float(ax1), float(e["angle"])
    except Exception:
        return None


def normalize_coords(x: float, y: float, ellipse: tuple):
    """
    Convert pixel (x, y) to hoop-normalized coordinates using the ellipse.

    OpenCV fitEllipse returns (axes[0], axes[1], angle) where angle is the
    rotation of axes[0] from horizontal. axes[0] is not necessarily the major
    axis. We identify the major axis by size, derive its angle, then rotate so
    the major axis becomes vertical and normalize by the semi-axes.
    """
    cx, cy, ax0, ax1, stored_angle = ellipse

    if ax0 >= ax1:
        major, minor, major_angle = ax0, ax1, stored_angle
    else:
        major, minor, major_angle = ax1, ax0, stored_angle - 90.0

    dx = x - cx
    dy = y - cy

    theta = math.radians(90.0 - major_angle)
    xr = dx * math.cos(theta) - dy * math.sin(theta)
    yr = dx * math.sin(theta) + dy * math.cos(theta)

    xn = xr / (minor / 2)
    yn = yr / (major / 2)
    dist_n = math.sqrt(xn ** 2 + yn ** 2)

    return round(xn, 4), round(yn, 4), round(dist_n, 4)


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
            "diameter_px": round(float(radius) * 2, 2),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Extract per-frame ball tracking data into data/ball_tracking/."
    )
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract even if a CSV already exists for a shot")
    ap.add_argument("--batch", metavar="BATCH_ID",
                    help="Limit processing to a specific batch (e.g. pending_1826811162)")
    args = ap.parse_args()

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    shots = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "skip":
                continue
            shots.append(row)

    print(f"Processing {len(shots)} shots\n")

    written          = 0
    written_norm     = 0
    skipped          = 0
    no_ball          = 0

    for shot in shots:
        dataset_name = shot["dataset_name"]
        shot_dir     = REPO_ROOT / shot["rel_shot_dir"]
        parts        = Path(shot["rel_shot_dir"]).parts
        batch_id     = next((p for p in parts if p.startswith(("pending_", "live_"))), "unknown")
        out_path      = TRACKING_DIR   / f"{batch_id}_{dataset_name}.csv"
        out_path_norm = NORMALIZED_DIR / f"{batch_id}_{dataset_name}.csv"

        if args.batch and batch_id != args.batch:
            continue

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
            writer = csv.DictWriter(f, fieldnames=["frame_index", "x", "y", "radius", "diameter_px"])
            writer.writeheader()
            writer.writerows(centers)

        print(f"  wrote {len(centers):4d} frames  {dataset_name}")
        written += 1

        ellipse = load_ellipse(shot.get("ellipse_meta", ""))
        if ellipse is None:
            print(f"  WARNING: no ellipse for {dataset_name} — normalized file not written")
            continue

        _, _, ax0, ax1, _ = ellipse
        minor = min(ax0, ax1)   # full minor axis length in pixels

        norm_rows = []
        for row in centers:
            xn, yn, dist_n = normalize_coords(row["x"], row["y"], ellipse)
            diameter_norm  = round(row["diameter_px"] / minor, 4)
            norm_rows.append({
                "frame_index":  row["frame_index"],
                "xn":           xn,
                "yn":           yn,
                "dist_n":       dist_n,
                "diameter_norm": diameter_norm,
            })

        with open(out_path_norm, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_index", "xn", "yn", "dist_n", "diameter_norm"])
            writer.writeheader()
            writer.writerows(norm_rows)

        written_norm += 1

    print(f"\nDone. written={written}  normalized={written_norm}  skipped={skipped}  no_ball={no_ball}")


if __name__ == "__main__":
    main()
