import csv
import json
import cv2
import numpy as np
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent
CSV_PATH   = REPO_ROOT / "data" / "shot_labels.csv"
TARGET_DATE = "2026-03-16"

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


def main() -> None:
    # Load all shots from the target date
    affected = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date = row["created_at"][:10]
            if date == TARGET_DATE and row["ellipse_meta"]:
                affected.append(row)

    if not affected:
        print(f"No shots found for {TARGET_DATE}.")
        return

    print(f"Found {len(affected)} shots from {TARGET_DATE}.")

    # Open a frame from the first affected shot
    first_frame_path = find_first_frame(affected[0]["rel_shot_dir"])
    if first_frame_path is None:
        print("No frames found for the first affected shot.")
        return

    frame = cv2.imread(str(first_frame_path))
    if frame is None:
        print(f"Could not load frame: {first_frame_path}")
        return

    cv2.namedWindow("Fit Ellipse")
    cv2.setMouseCallback("Fit Ellipse", mouse_callback, {"frame": frame})

    print("Click points around the hoop rim (min 5).")
    print("Left click: add point  |  Right click: undo  |  Enter: confirm")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(points) >= 5:  # Enter
            break
        if key == 27:  # Escape
            print("Cancelled.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    pts     = np.array(points, dtype=np.float32)
    ellipse = cv2.fitEllipse(pts)
    (cx, cy), (major, minor), angle = ellipse

    ellipse_data = {
        "ellipse": {
            "center": [float(cx), float(cy)],
            "axes":   [float(major), float(minor)],
            "angle":  float(angle),
        }
    }

    print(f"\nFitted ellipse: center=({cx:.1f}, {cy:.1f}), axes=({major:.1f}, {minor:.1f}), angle={angle:.1f}")

    # Write to all affected ellipse JSON files
    updated = 0
    for row in affected:
        ellipse_path = REPO_ROOT / row["ellipse_meta"]
        ellipse_path.parent.mkdir(parents=True, exist_ok=True)
        ellipse_path.write_text(json.dumps(ellipse_data, indent=2), encoding="utf-8")
        updated += 1

    print(f"Updated {updated} ellipse files.")


if __name__ == "__main__":
    main()
