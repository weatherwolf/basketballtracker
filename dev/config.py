"""
config.py

Central configuration for all dev scripts.
Import with:
    from config import REPO_ROOT, LABELS_CSV, ...
    (from utils/: sys.path.insert(0, str(Path(__file__).resolve().parent.parent)))
"""

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
LABELS_JSON         = REPO_ROOT / "data" / "shot_labels.json"
LABELS_CSV          = REPO_ROOT / "data" / "shot_labels.csv"
TRACKING_DIR        = REPO_ROOT / "data" / "ball_tracking"
NORMALIZED_DIR      = REPO_ROOT / "data" / "ball_tracking_normalized"
STICKER_TRACKING_DIR   = REPO_ROOT / "data" / "sticker_tracking"
STICKER_MORPH_CLOSE_PX = 25  # kernel size for closing gaps in sticker HSV mask (e.g. net occlusion)

# ---------------------------------------------------------------------------
# Work paths
# ---------------------------------------------------------------------------
WORK_DIR            = REPO_ROOT / "work"
RUNS_DIR            = WORK_DIR / "runs"
FRAMES_BATCH_DIR    = WORK_DIR / "frames_batch"

# ---------------------------------------------------------------------------
# Asset / media paths
# ---------------------------------------------------------------------------
GLOBAL_ELLIPSE      = REPO_ROOT / "assets" / "hoop_ellipses.json"
ELLIPSES_DIR        = REPO_ROOT / "assets" / "hoop_ellipses"
EXPORTS_DIR         = REPO_ROOT / "media" / "exports"

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
MODEL_PATH          = REPO_ROOT / "raspberry" / "minirocket_model.joblib"

# ---------------------------------------------------------------------------
# HSV thresholds for ball (orange) detection
# Note: raspberry/inference.py uses slightly looser values [5,150,100]
# ---------------------------------------------------------------------------
LOWER_ORANGE = np.array([5, 120, 120])
UPPER_ORANGE = np.array([20, 255, 255])

# ---------------------------------------------------------------------------
# ML constants
# ---------------------------------------------------------------------------
DIST_THRESHOLD   = 1.1   # max normalised ball-to-hoop distance to include a shot
MIN_DIAMETER_NORM = 0.38 # minimum normalised ball diameter to accept a detection (filters false positives)
SERIES_LENGTH   = 50    # number of timesteps after resampling
CHANNELS        = ["xn", "yn", "dist_n", "diameter_norm",
                   "vx", "vy", "v_dist", "v_diam",
                   "ax", "ay", "a_dist", "a_diam"]

# ---------------------------------------------------------------------------
# shot_labels.csv field order (used by any writer of this file)
# ---------------------------------------------------------------------------
LABEL_CSV_FIELDS = [
    "dataset_name", "rel_shot_dir", "label", "rel_preview_mp4",
    "rel_export_mp4", "ellipse_meta", "notes", "created_at", "updated_at", "has_stickers",
    "has_8_stickers",
]
