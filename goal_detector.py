# goal_detector.py
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# --- folders (edit here) ---
REPO_ROOT = Path(__file__).resolve().parent
WORK_DIR = REPO_ROOT / "work"
DATA_DIR = REPO_ROOT / "data"
FRAMES_DIR = WORK_DIR / "frames_batch"
OUT_CSV_DIR = DATA_DIR / "ball_tracking"
DEBUG_DIR = WORK_DIR / "debug" / "goal_detector"
ELLIPSE_META_DIR = OUT_CSV_DIR  # simplest: store alongside CSVs

# --- ball color threshold ---
LOWER_ORANGE = np.array([5, 120, 120])
UPPER_ORANGE = np.array([20, 255, 255])

# --- detection knobs ---
MAX_FRAMES_TO_EXIT = 12

# --- hoop model (fallback/default) ---
ELLIPSE = ((268.76812744140625, 224.77842712402344), (43.57143783569336, 108.83766174316406), 87.49404907226562)

# --- supported image extensions ---
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --- flags ---
DEBUG_FLAG_PRINT = True


def _ellipse_meta_path(frame_name: str) -> Path:
    return ELLIPSE_META_DIR / f"ellipse_{frame_name}.json"


def save_ellipse_for_dataset(frame_name: str, ellipse_tuple) -> Path:
    (cx, cy), (major, minor), angle = ellipse_tuple
    data = {
        "ellipse": {
            "center": [float(cx), float(cy)],
            "axes": [float(major), float(minor)],
            "angle": float(angle),
        }
    }
    path = _ellipse_meta_path(frame_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def load_ellipse_for_dataset(frame_name: str, default_ellipse):
    path = _ellipse_meta_path(frame_name)
    if not path.exists():
        return default_ellipse

    try:
        data = json.loads(path.read_text())
        e = data["ellipse"]
        cx, cy = e["center"]
        major, minor = e["axes"]
        angle = e["angle"]
        return ((cx, cy), (major, minor), angle)
    except Exception:
        return default_ellipse


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def _parse_frame_filename(filename: str):
    stem = Path(filename).stem
    m = re.match(r"^(.*)_(\d+)$", stem)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def point_inside_ellipse(x: float, y: float, ellipse_tuple) -> bool:
    (cx, cy), (major, minor), angle = ellipse_tuple

    axes = (int(round(major / 2.0)), int(round(minor / 2.0)))
    center = (int(round(cx)), int(round(cy)))
    ang = int(round(angle))

    poly = cv2.ellipse2Poly(center, axes, ang, 0, 360, 5)
    poly = poly.reshape((-1, 1, 2))

    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def _find_ball_center(img_bgr: np.ndarray):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    return float(x), float(y), float(radius)


def _write_debug_frame(
    img_bgr: np.ndarray,
    out_path: Path,
    ellipse_tuple,
    x,
    y,
    radius,
    inside
):
    out = img_bgr.copy()

    cv2.ellipse(out, ellipse_tuple, (0, 255, 0), 2)

    if x is not None and y is not None and radius is not None:
        cv2.circle(out, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)

    label = "INSIDE" if inside == 1 else "OUTSIDE"
    cv2.putText(
        out,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def detect_goal_state_machine(rows, hoop_center_y, max_frames_to_exit=12):
    state = 0
    entry_frame = None
    inside_below_frame = None

    ever_inside = False
    ever_inside_above = False
    ever_inside_below = False
    ever_exit_below = False

    for frame_idx, x, y, inside in rows:
        if x is None or y is None:
            continue

        above = y < hoop_center_y
        below = y > hoop_center_y

        if inside == 1:
            ever_inside = True
            if above:
                ever_inside_above = True
            if below:
                ever_inside_below = True

        if state == 0:
            if inside == 1 and above:
                state = 1
                entry_frame = frame_idx

        elif state == 1:
            if inside == 1 and below:
                state = 2
                inside_below_frame = frame_idx
            elif inside == 0:
                state = 0
                entry_frame = None

        elif state == 2:
            if (frame_idx - inside_below_frame) > max_frames_to_exit:
                state = 0
                entry_frame = None
                inside_below_frame = None
                continue

            if inside == 0 and below:
                ever_exit_below = True
                return True, entry_frame, inside_below_frame, frame_idx, None

            if inside == 1 and above:
                state = 0
                entry_frame = None
                inside_below_frame = None

    if not ever_inside:
        reason = "MISS: ball was never inside the hoop ellipse"
    elif not ever_inside_above:
        reason = "MISS: ball entered hoop region but never from above"
    elif not ever_inside_below:
        reason = "MISS: ball entered from above but never went below rim center"
    elif not ever_exit_below:
        reason = "MISS: ball went below rim but never exited downward (likely rim bounce)"
    else:
        reason = "MISS: unknown temporal ordering issue"

    return False, None, None, None, reason


def _collect_groups(frames_dir: Path):
    groups = defaultdict(list)

    # Walk recursively so frames can live in per-shot subfolders.
    for root, _dirs, files in os.walk(frames_dir):
        r = Path(root)
        for f in files:
            p = r / f
            if not p.is_file() or not _is_image_file(p):
                continue

            frame_name, frame_idx = _parse_frame_filename(p.name)
            if frame_name is None:
                continue

            groups[frame_name].append((frame_idx, p))

    for k in groups:
        groups[k].sort(key=lambda t: t[0])

    return groups


def _write_group_csv(frame_name, rows, out_csv_dir):
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_csv_dir / f"centers_{frame_name}.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_name", "frame_index", "x", "y", "radius", "inside_ellipse"])
        for r in rows:
            writer.writerow(
                [frame_name, r["frame_index"], r["x"], r["y"], r["radius"], r["inside"]]
            )

    return csv_path


def main():
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {FRAMES_DIR}")

    OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    groups = _collect_groups(FRAMES_DIR)
    if not groups:
        print("No matching frame files found.")
        return

    summary = []

    for frame_name, items in sorted(groups.items()):
        dataset_ellipse = load_ellipse_for_dataset(frame_name, ELLIPSE)
        hoop_center_y = float(dataset_ellipse[0][1])

        debug_subdir = DEBUG_DIR / frame_name
        group_rows_for_csv = []
        rows_for_logic = []

        for frame_idx, path in items:
            img = cv2.imread(str(path))

            if img is None:
                group_rows_for_csv.append(
                    {"frame_index": frame_idx, "x": None, "y": None, "radius": None, "inside": 0}
                )
                continue

            x, y, radius = _find_ball_center(img)
            inside = int(point_inside_ellipse(x, y, dataset_ellipse)) if x is not None else 0

            group_rows_for_csv.append(
                {"frame_index": frame_idx, "x": x, "y": y, "radius": radius, "inside": inside}
            )
            rows_for_logic.append((frame_idx, x, y, inside))

            _write_debug_frame(
                img,
                debug_subdir / path.name,
                dataset_ellipse,
                x,
                y,
                radius,
                inside
            )

        goal, entry_f, inside_below_f, exit_f, miss_reason = detect_goal_state_machine(
            rows_for_logic,
            hoop_center_y,
            MAX_FRAMES_TO_EXIT
        )

        csv_path = _write_group_csv(frame_name, group_rows_for_csv, OUT_CSV_DIR)
        summary.append((frame_name, goal, entry_f, inside_below_f, exit_f, miss_reason, csv_path))

    print("\n--- Summary ---")
    for frame_name, goal, entry_f, inside_below_f, exit_f, miss_reason, _ in summary:
        if goal:
            print(
                f"{frame_name}: GOAL detected ✅  entry={entry_f}, "
                f"inside_below={inside_below_f}, exit_below={exit_f}"
            )
        else:
            print(f"{frame_name}: No GOAL detected.")
            if DEBUG_FLAG_PRINT and miss_reason is not None:
                print("Reason:", miss_reason)

    print(f"\nDebug frames written under: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
