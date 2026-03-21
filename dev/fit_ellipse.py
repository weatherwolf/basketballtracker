import argparse
import csv
import json
import cv2
import numpy as np
from datetime import date
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
CSV_PATH   = REPO_ROOT / "data" / "shot_labels.csv"
GLOBAL_ELLIPSE = REPO_ROOT / "assets" / "hoop_ellipses.json"

points = []


def mouse_callback(event, x, y, flags, param):
    frame_copy = param["frame"].copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()

    for pt in points:
        cv2.circle(frame_copy, pt, 4, (0, 255, 0), -1)

    if len(points) >= 5:
        pts = np.array(points, dtype=np.float32)
        ellipse = cv2.fitEllipse(pts)
        cv2.ellipse(frame_copy, ellipse, (0, 0, 255), 2)

    cv2.putText(frame_copy, f"Points: {len(points)}  |  Left click: add  Right click: undo  Enter: confirm",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Fit Ellipse", frame_copy)


def find_first_frame(rel_shot_dir: str) -> Path | None:
    shot_dir = REPO_ROOT / rel_shot_dir
    frames = sorted(f for f in shot_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return frames[0] if frames else None


def load_existing_ellipse(ellipse_meta: str) -> tuple | None:
    """Return cv2-style ellipse tuple ((cx,cy),(major,minor),angle) or None."""
    path = REPO_ROOT / ellipse_meta
    if not path.exists():
        # fall back to global default
        if GLOBAL_ELLIPSE.exists():
            path = GLOBAL_ELLIPSE
        else:
            return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        e = data["ellipse"]
        cx, cy = e["center"]
        major, minor = e["axes"]
        return (float(cx), float(cy)), (float(major), float(minor)), float(e["angle"])
    except Exception:
        return None


def verify_ellipse(frame: np.ndarray, ellipse: tuple, title: str) -> bool:
    """Show frame with ellipse drawn. Return True if user confirms it looks correct."""
    preview = frame.copy()
    (cx, cy), (major, minor), angle = ellipse
    cv2.ellipse(preview, ((int(cx), int(cy)), (int(major), int(minor)), angle), (0, 255, 0), 2)
    cv2.circle(preview, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    cv2.putText(preview, f"{title}  |  Press Y to accept, N to refit",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Fit Ellipse", preview)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('y'), ord('Y')):
            cv2.destroyAllWindows()
            return True
        if key in (ord('n'), ord('N'), 27):
            cv2.destroyAllWindows()
            return False


def interactive_fit(frame: np.ndarray) -> tuple | None:
    """Let user click points and fit an ellipse. Returns cv2 ellipse tuple or None."""
    points.clear()
    cv2.namedWindow("Fit Ellipse")
    cv2.setMouseCallback("Fit Ellipse", mouse_callback, {"frame": frame})
    # Draw initial blank frame
    cv2.imshow("Fit Ellipse", frame)
    print("Click points around the hoop rim (min 5).")
    print("Left click: add  |  Right click: undo  |  Enter: confirm  |  Escape: cancel")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(points) >= 5:  # Enter
            break
        if key == 27:  # Escape
            print("Cancelled.")
            cv2.destroyAllWindows()
            return None
    cv2.destroyAllWindows()
    pts = np.array(points, dtype=np.float32)
    return cv2.fitEllipse(pts)


def save_ellipse(ellipse: tuple, affected: list[dict]) -> None:
    (cx, cy), (major, minor), angle = ellipse
    ellipse_data = {
        "ellipse": {
            "center": [float(cx), float(cy)],
            "axes":   [float(major), float(minor)],
            "angle":  float(angle),
        }
    }
    print(f"\nFitted ellipse:\n{json.dumps(ellipse_data, indent=2)}")
    updated = 0
    for row in affected:
        ellipse_path = REPO_ROOT / row["ellipse_meta"]
        ellipse_path.parent.mkdir(parents=True, exist_ok=True)
        ellipse_path.write_text(json.dumps(ellipse_data, indent=2), encoding="utf-8")
        updated += 1
    print(f"Updated {updated} ellipse files.")


def load_shots(*, batch_id: str | None, target_date: str | None) -> list[dict]:
    affected = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["ellipse_meta"]:
                continue
            if batch_id:
                if batch_id in row["rel_shot_dir"]:
                    affected.append(row)
            elif target_date:
                if row["created_at"][:10] == target_date:
                    affected.append(row)
    return affected


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--batch", metavar="BATCH_ID",
                       help="Fit ellipse for all shots in this batch (e.g. pending_1826811162)")
    group.add_argument("--date", metavar="YYYY-MM-DD", default=str(date.today()),
                       help="Fit ellipse for all shots created on this date (default: today)")
    ap.add_argument("--manual-check", action="store_true",
                    help="Use work/inputs/camera_check.mp4 instead of shot data; saves to global ellipse file")
    args = ap.parse_args()

    if args.manual_check:
        mp4_path = REPO_ROOT / "work" / "inputs" / "camera_check.mp4"
        if not mp4_path.exists():
            print(f"Not found: {mp4_path}")
            print("Run manual_camera_check.bat first to download the video.")
            return
        cap = cv2.VideoCapture(str(mp4_path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"Could not read frame from: {mp4_path}")
            return
        print(f"Using frame from: {mp4_path}")
        ellipse = interactive_fit(frame)
        if ellipse is None:
            return
        (cx, cy), (major, minor), angle = ellipse
        ellipse_data = {
            "ellipse": {
                "center": [float(cx), float(cy)],
                "axes":   [float(major), float(minor)],
                "angle":  float(angle),
            }
        }
        print(f"\nFitted ellipse:\n{json.dumps(ellipse_data, indent=2)}")
        GLOBAL_ELLIPSE.parent.mkdir(parents=True, exist_ok=True)
        GLOBAL_ELLIPSE.write_text(json.dumps(ellipse_data, indent=2), encoding="utf-8")
        print(f"Saved to: {GLOBAL_ELLIPSE}")
        return

    if args.batch:
        affected = load_shots(batch_id=args.batch, target_date=None)
        label = f"batch {args.batch}"
    else:
        affected = load_shots(batch_id=None, target_date=args.date)
        label = f"date {args.date}"

    if not affected:
        print(f"No shots found for {label}.")
        return

    print(f"Found {len(affected)} shots for {label}.")

    first_frame_path = find_first_frame(affected[0]["rel_shot_dir"])
    if first_frame_path is None:
        print("No frames found for the first shot.")
        return

    frame = cv2.imread(str(first_frame_path))
    if frame is None:
        print(f"Could not load frame: {first_frame_path}")
        return

    ellipse = interactive_fit(frame)
    if ellipse is None:
        return

    save_ellipse(ellipse, affected)


if __name__ == "__main__":
    main()
