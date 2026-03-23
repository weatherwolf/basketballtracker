import argparse
import csv
import json
import cv2
import numpy as np
from datetime import date
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT, LABELS_CSV as CSV_PATH, GLOBAL_ELLIPSE

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


def detect_sticker_centers(frame: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[tuple[float, float]]:
    """Return (x, y) centroid of each sticker blob found in frame via HSV mask."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    # Small open/close to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centers.append((cx, cy))
    return centers


def sticker_fit(frame: np.ndarray, lo: np.ndarray, hi: np.ndarray, show_debug: bool = True) -> tuple | None:
    """Detect sticker centers, fit ellipse, show debug image. Returns cv2 ellipse tuple or None."""
    centers = detect_sticker_centers(frame, lo, hi)
    print(f"Detected {len(centers)} sticker(s): {[(round(x,1), round(y,1)) for x, y in centers]}")

    if len(centers) != 8:
        print(f"Expected 8 stickers, found {len(centers)}. Tune HSV args.")
        debug = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_vis = cv2.inRange(hsv, lo, hi)
        for (x, y) in centers:
            cv2.circle(debug, (int(x), int(y)), 8, (255, 0, 255), -1)
            cv2.circle(debug, (int(x), int(y)), 9, (0, 0, 0), 1)
        cv2.putText(debug, f"Found {len(centers)} stickers — expected 8  |  Press any key",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        side_by_side = np.hstack([debug, mask_bgr])
        cv2.imshow("Sticker Detection — ERROR", side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    pts = np.array(centers, dtype=np.float32)
    ellipse = cv2.fitEllipse(pts)
    (cx, cy), (major, minor), angle = ellipse
    if show_debug:
        debug = frame.copy()
        for (x, y) in centers:
            cv2.circle(debug, (int(x), int(y)), 8, (255, 0, 255), -1)
            cv2.circle(debug, (int(x), int(y)), 9, (0, 0, 0), 1)
        cv2.ellipse(debug, ellipse, (0, 255, 0), 2)
        cv2.circle(debug, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(debug, f"{len(centers)} stickers  |  Press any key to continue",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(debug, f"center=({cx:.1f},{cy:.1f})  axes=({major:.1f},{minor:.1f})  angle={angle:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.imshow("Sticker Detection", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return ellipse


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--batch", metavar="BATCH_ID",
                       help="Fit ellipse for all shots in this batch (e.g. pending_1826811162)")
    group.add_argument("--date", metavar="YYYY-MM-DD", default=str(date.today()),
                       help="Fit ellipse for all shots created on this date (default: today)")
    ap.add_argument("--manual-check", action="store_true",
                    help="Use work/inputs/camera_check.mp4 instead of shot data; saves to global ellipse file")
    ap.add_argument("--sticker-check", action="store_true",
                    help="Auto-fit ellipse from sticker HSV detection on camera_check.mp4; saves to global ellipse file")
    ap.add_argument("--sticker-check-silent", action="store_true",
                    help="Same as --sticker-check but skips the debug image display")
    # HSV tuning (same defaults as test_sticker_hsv.py)
    ap.add_argument("--h-center", type=int, default=54,  help="Sticker hue centre OpenCV 0-179 (default 54)")
    ap.add_argument("--h-range",  type=int, default=15,  help="+-hue range around centre (default 15)")
    ap.add_argument("--s-lo",     type=int, default=55,  help="Min saturation 0-255 (default 55)")
    ap.add_argument("--s-hi",     type=int, default=175, help="Max saturation 0-255 (default 175)")
    ap.add_argument("--v-lo",     type=int, default=60,  help="Min value 0-255 (default 60)")
    ap.add_argument("--v-hi",     type=int, default=190, help="Max value 0-255 (default 190)")
    args = ap.parse_args()

    if args.sticker_check or args.sticker_check_silent or args.manual_check:
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

        if args.sticker_check or args.sticker_check_silent:
            h_lo = max(0,   args.h_center - args.h_range)
            h_hi = min(179, args.h_center + args.h_range)
            lo = np.array([h_lo, args.s_lo, args.v_lo], dtype=np.uint8)
            hi = np.array([h_hi, args.s_hi, args.v_hi], dtype=np.uint8)
            print(f"HSV range: H[{h_lo}-{h_hi}] S[{args.s_lo}-{args.s_hi}] V[{args.v_lo}-{args.v_hi}]")
            ellipse = sticker_fit(frame, lo, hi, show_debug=not args.sticker_check_silent)
        else:
            ellipse = interactive_fit(frame)

        if ellipse is None:
            sys.exit(1)
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
