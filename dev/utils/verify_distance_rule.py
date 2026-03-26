"""
Verify that shots where the minimum ball-to-hoop distance across all frames
is >= --dist are always misses.

Reads pre-computed ball tracking CSVs from data/ball_tracking/ instead of
re-detecting from frames. Run extract_ball_tracking.py first.

Prints any violations found and a final pass/fail summary.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT, LABELS_CSV as CSV_PATH, TRACKING_DIR, NORMALIZED_DIR, DIST_THRESHOLD
from delete_clip import delete_clip
from reverse_label import flip_label


def load_ellipse(ellipse_meta_rel: str):
    path = REPO_ROOT / ellipse_meta_rel
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    e = data["ellipse"]
    return float(e["center"][0]), float(e["center"][1])


def min_ball_distance(tracking_csv: Path, hoop_cx: float, hoop_cy: float):
    min_dist = float("inf")
    with open(tracking_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dist = math.hypot(float(row["x"]) - hoop_cx, float(row["y"]) - hoop_cy)
            if dist < min_dist:
                min_dist = dist
    return min_dist if min_dist < float("inf") else None


def min_normalized_distance(tracking_csv: Path):
    """Read the pre-computed dist_n column from a normalized tracking CSV."""
    min_dist = float("inf")
    with open(tracking_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dist = float(row["dist_n"])
            if dist < min_dist:
                min_dist = dist
    return min_dist if min_dist < float("inf") else None


def open_video(path: Path) -> None:
    os.startfile(str(path))

def main():
    ap = argparse.ArgumentParser(
        description="Verify that shots with min ball-to-hoop distance >= DIST are always misses."
    )
    ap.add_argument("--dist", type=float, default=None,
                    help="Distance threshold (default: 35 in pixel mode, 1.0 in normalized mode)")
    ap.add_argument("--normalized", action="store_true",
                    help="Use hoop-normalized coordinates from data/ball_tracking_normalized/")
    ap.add_argument("--show-violations", action="store_true",
                    help="Open show_closest_frames.py for each violation shot")
    ap.add_argument("--show-violations-video", action="store_true",
                    help="Open show_closest_frames.py for each violation shot")
    ap.add_argument("--only-live", action="store_true",
                    help="Only check shots from live_ batches")
    args = ap.parse_args()

    if args.normalized:
        tracking_dir = NORMALIZED_DIR
        threshold    = args.dist if args.dist is not None else 1.0
        dist_unit    = "norm"
    else:
        tracking_dir = TRACKING_DIR
        threshold    = args.dist if args.dist is not None else 35.0
        dist_unit    = "px"

    shots = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "skip" or not row["ellipse_meta"]:
                continue
            parts    = Path(row["rel_shot_dir"]).parts
            batch_id = next((p for p in parts if p.startswith(("pending_", "live_"))), None)
            if not batch_id:
                continue
            if args.only_live and not batch_id.startswith("live_"):
                continue
            shots.append(row)

    print(f"Checking {len(shots)} shots (threshold = {threshold}{dist_unit})\n")

    violations    = []
    skipped       = []
    furthest_make = 0
    within_threshold = 0
    total_makes = 0
    total_misses = 0

    for shot in shots:
        dataset_name = shot["dataset_name"]
        label        = shot["label"]

        parts    = Path(shot["rel_shot_dir"]).parts
        batch_id = next((p for p in parts if p.startswith(("pending_", "live_"))), "unknown")
        tracking_csv = tracking_dir / f"{batch_id}_{dataset_name}.csv"

        if not tracking_csv.exists():
            skipped.append(f"{batch_id}/{dataset_name}: tracking CSV missing ({tracking_csv.name})")
            continue

        if args.normalized:
            min_dist = min_normalized_distance(tracking_csv)
        else:
            ellipse = load_ellipse(shot["ellipse_meta"])
            if ellipse is None:
                skipped.append(f"{batch_id}/{dataset_name}: ellipse file missing")
                continue
            hoop_cx, hoop_cy = ellipse
            min_dist = min_ball_distance(tracking_csv, hoop_cx, hoop_cy)

        if min_dist is None:
            skipped.append(f"{batch_id}/{dataset_name}: no rows in tracking CSV")
            continue

        rule_says_miss = min_dist >= threshold
        if rule_says_miss and label != "miss":
            msg = f"  VIOLATION  {batch_id}/{dataset_name}  label={label.upper()}  min_dist={min_dist:.3f}{dist_unit}"
            violations.append(msg)
            print(msg)
            if args.show_violations:
                cmd = [sys.executable, str(Path(__file__).parent / "show_closest_frames.py"),
                       "--shot", dataset_name, "--batch", batch_id]
                if not args.normalized:
                    cmd += ["--dist", str(threshold)]
                subprocess.run(cmd)
            elif args.show_violations_video:
                video = shot.get("rel_preview_mp4") or shot.get("rel_export_mp4")
                if not video:
                    print(f"  (no video found for {batch_id}/{dataset_name})")
                else:
                    preview_path = REPO_ROOT / video
                    if not preview_path.exists():
                        print(f"  (video file missing: {video})")
                    else:
                        open_video(preview_path)
                        try:
                            raw = input("  Press Enter for next, r to reverse label, d to delete, q to quit: ").strip().lower()
                        except EOFError:
                            break
                        if raw == "q":
                            break
                        if raw == "r":
                            flip_label(batch_id, dataset_name)
                        if raw == "d":
                            try:
                                confirm = input(f"  Confirm delete {batch_id} / {dataset_name}? [y/N]: ").strip().lower()
                            except EOFError:
                                confirm = ""
                            if confirm == "y":
                                delete_clip(batch_id, dataset_name)
                            else:
                                print("  Aborted.")

        else:
            status = f"{'MISS (rule)' if rule_says_miss else 'close shot  ':11s}"
            print(f"  ok  {status}  {batch_id}/{dataset_name:40s}  min_dist={min_dist:.3f}{dist_unit}  label={label}")

        if label == "goal" and min_dist > furthest_make:
            furthest_make = min_dist
        if not rule_says_miss:
            within_threshold += 1
            if label == "goal":
                total_makes+=1
            elif label == "miss":
                total_misses+=1

    assert total_makes + total_misses == within_threshold, f"other data types leaked into within_threshold {total_makes} + {total_misses} != {within_threshold}"

    print()
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")
        print()

    if violations:
        print(f"\nRULE VIOLATIONS ({len(violations)}) — rule does NOT always hold.")
        for v in violations:
            print(v)
        sys.exit(1)
    else:
        print(f"PASSED — rule holds across all {len(shots) - len(skipped)} checked shots.")
        print(f"Furthest make: {furthest_make:.3f}{dist_unit}")
        print(f"Shots within threshold: {within_threshold}, Makes: {total_makes}, Misses: {total_misses}")


if __name__ == "__main__":
    main()
