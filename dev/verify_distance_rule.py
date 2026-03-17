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
import sys
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent.parent
CSV_PATH      = REPO_ROOT / "data" / "shot_labels.csv"
TRACKING_DIR  = REPO_ROOT / "data" / "ball_tracking"


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
        batch_id = next((p for p in parts if p.startswith("pending_")), "unknown")
        tracking_csv = TRACKING_DIR / f"{batch_id}_{dataset_name}.csv"

        if not tracking_csv.exists():
            skipped.append(f"{dataset_name}: tracking CSV missing ({tracking_csv.name})")
            continue

        ellipse = load_ellipse(shot["ellipse_meta"])
        if ellipse is None:
            skipped.append(f"{dataset_name}: ellipse file missing")
            continue

        hoop_cx, hoop_cy = ellipse
        min_dist = min_ball_distance(tracking_csv, hoop_cx, hoop_cy)

        if min_dist is None:
            skipped.append(f"{dataset_name}: no rows in tracking CSV")
            continue

        rule_says_miss = min_dist >= threshold
        if rule_says_miss and label != "miss":
            violations.append(
                f"  VIOLATION  {dataset_name}  label={label.upper()}  min_dist={min_dist:.1f}px"
            )
        else:
            status = f"{'MISS (rule)' if rule_says_miss else 'close shot  ':11s}"
            print(f"  ok  {status}  {dataset_name:40s}  min_dist={min_dist:.1f}px  label={label}")

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
        print(f"RULE VIOLATIONS ({len(violations)}) — rule does NOT always hold:")
        for v in violations:
            print(v)
        sys.exit(1)
    else:
        print(f"PASSED — rule holds across all {len(shots) - len(skipped)} checked shots.")
        print(f"Furthest make: {furthest_make:.1f}px")
        print(f"Shots within threshold: {within_threshold}, Makes: {total_makes}, Misses: {total_misses}")


if __name__ == "__main__":
    main()
