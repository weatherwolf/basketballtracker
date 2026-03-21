import argparse
import csv
import json
import math
import cv2
import numpy as np
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent
CSV_PATH     = REPO_ROOT / "data" / "shot_labels.csv"
TRACKING_DIR = REPO_ROOT / "data" / "ball_tracking"
TOP_N        = 5

LOWER_ORANGE = np.array([5, 120, 120])
UPPER_ORANGE = np.array([20, 255, 255])
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}


def load_ellipse(ellipse_meta_rel: str):
    path = REPO_ROOT / ellipse_meta_rel
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    e = data["ellipse"]
    cx, cy = e["center"]
    major, minor = e["axes"]
    angle = e["angle"]
    return (cx, cy), (major, minor), angle


def find_ball_center(img_bgr: np.ndarray):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    return float(x), float(y), float(radius)


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
        x, y, radius = find_ball_center(img)
        if x is None:
            continue
        # Parse frame index from filename (last numeric segment before extension)
        stem = path.stem
        parts = stem.rsplit("_", 1)
        frame_index = int(parts[-1]) if parts[-1].isdigit() else len(rows)
        rows.append({"frame_index": frame_index, "x": x, "y": y, "radius": radius, "path": path})
    return rows


def annotate_frame(img, ellipse, ball_x, ball_y, ball_radius, distance):
    out = img.copy()
    (cx, cy), (major, minor), angle = ellipse
    cv2.ellipse(out, ((cx, cy), (major, minor), angle), (0, 255, 0), 2)
    cv2.circle(out, (int(ball_x), int(ball_y)), int(ball_radius), (0, 165, 255), 2)
    cv2.circle(out, (int(ball_x), int(ball_y)), 3, (0, 0, 255), -1)
    cv2.putText(out, f"dist={distance:.1f}", (int(ball_x) + 8, int(ball_y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=float, default=None,
                    help="Show all frames with ball-to-hoop distance below this value, in temporal order")
    ap.add_argument("--show-frames", type=lambda v: v.lower() not in ("false", "0", "no"), default=True,
                    help="Flag to show the frames visually, default is True")
    ap.add_argument("--batch", metavar="BATCH_ID",
                    help="Only process shots from this batch (e.g. pending_1826811162)")
    ap.add_argument("--shot", metavar="SHOT_NAME",
                    help="Only process this specific shot by dataset_name (e.g. frame_test1_shot042)")
    args = ap.parse_args()

    shots = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "skip" or not row["ellipse_meta"]:
                continue
            if args.batch and args.batch not in row["rel_shot_dir"]:
                continue
            if args.shot and row["dataset_name"] != args.shot:
                continue
            shots.append(row)

    print(f"Processing {len(shots)} shots. Press any key to advance, Escape to quit.\n")

    for shot in shots:
        dataset_name = shot["dataset_name"]
        shot_dir     = REPO_ROOT / shot["rel_shot_dir"]
        ellipse      = load_ellipse(shot["ellipse_meta"])

        if ellipse is None:
            print(f"{dataset_name}: skipping (no ellipse)  [{shot_dir}]")
            continue

        if not shot_dir.exists():
            print(f"{dataset_name}: skipping (frames folder missing)  [{shot_dir}]")
            continue

        centers = compute_centers(shot_dir)
        if not centers:
            print(f"{dataset_name}: skipping (no ball detected in any frame)  [{shot_dir}]")
            continue

        cx, cy = ellipse[0]
        for row in centers:
            row["dist"] = math.hypot(row["x"] - cx, row["y"] - cy)

        if args.dist is not None:
            closest = sorted(
                (r for r in centers if r["dist"] <= args.dist),
                key=lambda r: r["frame_index"],
            )
        else:
            closest = sorted(centers, key=lambda r: r["dist"])[:TOP_N]

        annotated = []
        for row in closest:
            img = cv2.imread(str(row["path"]))
            if img is None:
                continue
            ann = annotate_frame(img, ellipse, row["x"], row["y"], row["radius"], row["dist"])
            annotated.append(ann)

        if not annotated:
            if args.dist is not None and shot["label"] == "goal":
                print(f"{dataset_name}: GOAL NOT WITHIN DISTANCE")
            else:
                print(f"{dataset_name}: no frames found within range")
            continue

        parts    = Path(shot["rel_shot_dir"]).parts
        batch_id = next((p for p in parts if p.startswith("pending_")), "unknown")
        header   = f"{batch_id}  |  {dataset_name}  [{shot['label'].upper()}]"

        if args.show_frames:
            if args.dist is not None:
                # Show frames one by one in temporal order
                quit_all = False
                for ann in annotated:
                    cv2.putText(ann, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Closest frames to hoop", ann)
                    key = cv2.waitKey(0) & 0xFF
                    if key == 27:
                        quit_all = True
                        break
                if quit_all:
                    break
            else:
                h = min(a.shape[0] for a in annotated)
                resized  = [cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h)) for a in annotated]
                combined = np.hstack(resized)
                cv2.putText(combined, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Closest frames to hoop", combined)
                key = cv2.waitKey(0) & 0xFF
                if key == 27:
                    break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
