"""
inference.py

Real-time shot classifier for Raspberry Pi.

Livestreams from the Pi camera, detects ball entry/exit, runs the MiniRocket
model on each clip, and displays GOAL / MISS in the terminal.

Expected files in the same directory:
    minirocket_model.joblib   (produced by: python dev/minirocket_test.py --save-model)
    ellipse.json              (copy of assets/hoop_ellipses.json)

Usage (from raspberry/ directory):
    python inference.py
    python inference.py --model minirocket_model.joblib --ellipse ellipse.json
"""

import json
import logging
import math
import select
import sys
import threading
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from libcamera import Transform
from picamera2 import Picamera2

sys.path.insert(0, str(Path(__file__).parent))
import shot_io

logging.getLogger("picamera2").setLevel(logging.ERROR)
logging.getLogger("libcamera").setLevel(logging.ERROR)

# --- camera / detection settings (matches livestream_filter.py) ---
LOWER_ORANGE     = np.array([5, 150, 100])
UPPER_ORANGE     = np.array([20, 255, 255])
MIN_BALL_PIXELS  = 50
FILTER_EVERY_N   = 4
DOWNSCALE_W, DOWNSCALE_H = 384, 216

# --- ball tracking HSV (matches extract_ball_tracking.py) ---
TRACK_LOWER = np.array([5, 120, 120])
TRACK_UPPER = np.array([20, 255, 255])

# --- ANSI ---
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CLEAR  = "\033[2J\033[H"

_showing_result  = False
_result_lock     = threading.Lock()
_quit_flag       = threading.Event()
_session: "shot_io.LabelSession | None" = None


def _key_listener() -> None:
    """Background thread: read single keypresses without requiring Enter.
    Uses cbreak mode (Ctrl+C still raises SIGINT).
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not _quit_flag.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch == "q":
                    _quit_flag.set()
                elif ch == "w":
                    if _session is not None:
                        result = _session.flip_last()
                        if result:
                            old_label, new_label = result
                            print(f"\n  {YELLOW}[w] {old_label} → {new_label}{RESET}", flush=True)
                        else:
                            print(f"\n  {YELLOW}[w] no shot to flip{RESET}", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _status(msg: str) -> None:
    if _showing_result:
        return
    print(f"\r  {msg:<70}", end="", flush=True)


def _show_result(label: str, confidence: float) -> None:
    global _showing_result
    _result_lock.acquire()
    _showing_result = True
    color = GREEN if label == "goal" else RED
    word  = "GOAL" if label == "goal" else "MISS"
    bar   = "█" * int(min(confidence, 3.0) / 3.0 * 32)

    print(CLEAR, end="")
    print(f"{color}{BOLD}")
    print(f"  ┌{'─' * 46}┐")
    print(f"  │{'':46}│")
    print(f"  │{word:^46}│")
    print(f"  │{'':46}│")
    print(f"  │  confidence  {confidence:6.3f}   {bar:<23}│")
    print(f"  │{'':46}│")
    print(f"  └{'─' * 46}┘")
    print(RESET, flush=True)
    time.sleep(3)
    print(CLEAR, end="", flush=True)
    _showing_result = False
    _result_lock.release()


def _show_no_shot(reason: str) -> None:
    global _showing_result
    _showing_result = True
    print(CLEAR, end="")
    print(f"{YELLOW}{BOLD}")
    print(f"  ┌{'─' * 46}┐")
    print(f"  │{'':46}│")
    print(f"  │{'no clear shot':^46}│")
    print(f"  │{reason:^46}│")
    print(f"  │{'':46}│")
    print(f"  └{'─' * 46}┘")
    print(RESET, flush=True)
    time.sleep(2)
    print(CLEAR, end="", flush=True)
    _showing_result = False


# --- preprocessing (mirrors minirocket_test.py exactly) ---

def _normalize_coords(x: float, y: float, ellipse: tuple):
    cx, cy, ax0, ax1, stored_angle = ellipse
    if ax0 >= ax1:
        major, minor, major_angle = ax0, ax1, stored_angle
    else:
        major, minor, major_angle = ax1, ax0, stored_angle - 90.0
    dx, dy = x - cx, y - cy
    theta  = math.radians(90.0 - major_angle)
    xr = dx * math.cos(theta) - dy * math.sin(theta)
    yr = dx * math.sin(theta) + dy * math.cos(theta)
    xn     = xr / (minor / 2)
    yn     = yr / (major / 2)
    dist_n = math.sqrt(xn ** 2 + yn ** 2)
    return xn, yn, dist_n


def _extract_centers(frames: list) -> list:
    """HSV ball detection on in-memory frames. Returns [(x, y, radius), ...]."""
    centers = []
    for frame in frames:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, TRACK_LOWER, TRACK_UPPER)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        centers.append((float(x), float(y), float(radius)))
    return centers


def _resample(sequence: np.ndarray, length: int) -> np.ndarray:
    n     = len(sequence)
    old_t = np.linspace(0, 1, n)
    new_t = np.linspace(0, 1, length)
    return np.stack(
        [np.interp(new_t, old_t, sequence[:, c]) for c in range(sequence.shape[1])],
        axis=1,
    )


def _add_derivatives(sequence: np.ndarray) -> np.ndarray:
    vel = np.gradient(sequence, axis=0)
    acc = np.gradient(vel,      axis=0)
    return np.concatenate([sequence, vel, acc], axis=1)


def _save_frames(frames: list, shot_dir: Path, dataset_name: str) -> None:
    """Write buffered frames to disk as JPEGs inside shot_dir."""
    shot_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(shot_dir / f"{dataset_name}_{i:06d}.jpg"), frame)


def run_inference(frames: list, model: dict, ellipse: tuple,
                  shot_dir: Path, dataset_name: str,
                  session: shot_io.LabelSession) -> None:
    global _last_shot_dir
    series_length  = model["series_length"]
    dist_threshold = model["dist_threshold"]
    rocket         = model["rocket"]
    scaler         = model["scaler"]
    clf            = model["clf"]

    _status("saving frames...")
    _save_frames(frames, shot_dir, dataset_name)
    _last_shot_dir = shot_dir

    _status("extracting ball centers...")
    centers = _extract_centers(frames)
    if not centers:
        _show_no_shot("no ball detected")
        return

    norm     = np.array([_normalize_coords(x, y, ellipse) for x, y, _ in centers])
    min_dist = norm[:, 2].min()
    if min_dist >= dist_threshold:
        _show_no_shot(f"ball too far  (min dist={min_dist:.2f})")
        return

    _status("running model...")
    resampled = _resample(norm, series_length)   # (series_length, 3)
    resampled = _add_derivatives(resampled)      # (series_length, 9)
    X         = resampled.T[np.newaxis]          # (1, 9, series_length)

    X_t        = rocket.transform(X)
    X_t        = scaler.transform(X_t)
    score      = clf.decision_function(X_t)[0]
    label      = "goal" if score >= 0 else "miss"
    confidence = abs(score)

    session.record(shot_dir, dataset_name, label)
    _show_result(label, confidence)


# --- camera helpers ---

def ball_present(frame_bgr: np.ndarray) -> bool:
    small = cv2.resize(frame_bgr, (DOWNSCALE_W, DOWNSCALE_H))
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    return cv2.countNonZero(mask) >= MIN_BALL_PIXELS


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Real-time shot classifier for Raspberry Pi.")
    ap.add_argument("--model",   default="minirocket_model.joblib", help="Path to model joblib file")
    ap.add_argument("--ellipse", default="ellipse.json",            help="Path to hoop ellipse JSON")
    args = ap.parse_args()

    import shutil
    _here     = Path(__file__).resolve().parent
    repo_root = _here.parent if _here.name == "raspberry" else _here
    for cleanup_root in [repo_root / "work" / "runs", repo_root / "media" / "exports"]:
        cleanup_root.mkdir(parents=True, exist_ok=True)
        for old in cleanup_root.iterdir():
            if old.is_dir() and old.name.startswith("live_"):
                print(f"Removing old live batch: {old.name}")
                shutil.rmtree(old)

    runs_root = repo_root / "work" / "runs"

    model_path   = Path(args.model)
    ellipse_path = Path(args.ellipse)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)
    if not ellipse_path.exists():
        print(f"Ellipse not found: {ellipse_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}...", end="", flush=True)
    model = joblib.load(model_path)
    print(" done")

    print(f"Loading ellipse from {ellipse_path}...", end="", flush=True)
    raw = json.loads(ellipse_path.read_text(encoding="utf-8"))
    e   = raw["ellipse"]
    ellipse = (
        float(e["center"][0]), float(e["center"][1]),
        float(e["axes"][0]),   float(e["axes"][1]),
        float(e["angle"]),
    )
    print(" done\n")

    # --- batch setup ---
    batch_id  = f"live_{time.strftime('%Y%m%d_%H%M%S')}"
    batch_dir = runs_root / batch_id / "frames_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Batch: {batch_id}")
    print(f"Saving shots to: {batch_dir}\n")

    (repo_root / "data").mkdir(parents=True, exist_ok=True)
    (repo_root / "media" / "exports").mkdir(parents=True, exist_ok=True)
    global _session
    _session = shot_io.LabelSession(repo_root=repo_root, batch_id=batch_id)
    session  = _session

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1536, 864), "format": "BGR888"},
        transform=Transform(hflip=True, vflip=True),
        controls={
            "FrameDurationLimits": (8333, 8333),
            "ScalerCrop": (1152, 648, 2304, 1296),  # 50% zoom
        },
    )
    picam2.configure(config)
    picam2.start()

    ball_in_frame = False
    frame_count   = 0
    frame_buffer  = []
    infer_thread  = None
    shot_count    = 0

    key_thread = threading.Thread(target=_key_listener, daemon=True)
    key_thread.start()

    _status(f"[{time.strftime('%H:%M:%S')}] ready — waiting for ball  (w=override  q=quit)")

    try:
        while not _quit_flag.is_set():
            frame = cv2.cvtColor(picam2.capture_array("main"), cv2.COLOR_RGB2BGR)
            frame_count += 1

            if ball_in_frame:
                frame_buffer.append(frame)

            if frame_count % FILTER_EVERY_N != 0:
                continue

            detected = ball_present(frame)

            if detected and not ball_in_frame:
                ball_in_frame = True
                frame_buffer  = [frame]
                _status(f"[{time.strftime('%H:%M:%S')}] ball in frame...")

            elif not detected and ball_in_frame:
                ball_in_frame = False
                shot_count   += 1
                n             = len(frame_buffer)
                buf           = frame_buffer.copy()
                frame_buffer  = []
                dataset_name  = f"shot_{shot_count:06d}"
                shot_dir      = batch_dir / dataset_name
                _status(f"[{time.strftime('%H:%M:%S')}] processing {n} frames...")
                infer_thread = threading.Thread(
                    target=run_inference,
                    args=(buf, model, ellipse, shot_dir, dataset_name, session),
                    daemon=True,
                )
                infer_thread.start()

    except KeyboardInterrupt:
        _quit_flag.set()
    finally:
        picam2.stop()
        for t in threading.enumerate():
            if t is not threading.current_thread() and t is not key_thread and t.is_alive():
                t.join(timeout=5.0)
        key_thread.join(timeout=1.0)
        print(f"\n  {YELLOW}saving labels...{RESET}", flush=True)
        n = session.save()
        print(f"  {YELLOW}session ended — {n} total shots in dataset{RESET}")


if __name__ == "__main__":
    main()
