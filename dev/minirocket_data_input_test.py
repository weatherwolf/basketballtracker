"""
minirocket_data_input_test.py

Tests how different input channel combinations affect MiniRocket accuracy.
A single final test set is held out first; all experiments train on the
remaining data and are evaluated on that same test set.

Run from repo root:
    python dev/minirocket_data_input_test.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LABELS_CSV, NORMALIZED_DIR, STICKER_TRACKING_DIR, DIST_THRESHOLD, MIN_DIAMETER_NORM, SERIES_LENGTH

# ---------------------------------------------------------------------------
# Channel index reference:
#   Ball (always present):
#     0=xn  1=yn  2=dist_n  3=diam
#     4=vx  5=vy  6=v_dist  7=v_diam
#     8=ax  9=ay  10=a_dist 11=a_diam
#   Sticker (only when include_stickers=True, requires has_8_stickers):
#     12=sticker_1 ... 19=sticker_8
# ---------------------------------------------------------------------------

EXPERIMENTS_ALL = [
    {"name": "distance only",            "ch": [2]},
    {"name": "position + distance",      "ch": [0, 1, 2]},
    {"name": "position + distance + diam", "ch": [0, 1, 2, 3]},
    {"name": "pos + dist + diam + velocity", "ch": [0, 1, 2, 3, 4, 5, 6, 7]},
]

EXPERIMENTS_ONLY_8_STICKERS = [
    # {"name": "distance only",                  "ch": [2]},
    # {"name": "position + distance",            "ch": [0, 1, 2]},
    # {"name": "position + distance + diam",     "ch": [0, 1, 2, 3]},
    # {"name": "pos + dist + diam + velocity",   "ch": [0, 1, 2, 3, 4, 5, 6, 7]},
    # {"name": "ball + stickers",                "ch": [0,1,2,3,4,5,6,7] + list(range(12, 20))},
    # {"name": "pos + dist + stickers",          "ch": [0, 1, 2] + list(range(12, 20))},
    {"name": "base + stickers",                "ch": [0, 1, 2, 3] + list(range(12, 20))},
]
TEST_FRACTION = 0.2
N_SEEDS       = 25
SEEDS         = list(range(N_SEEDS))


def load_all_channels(only_8_stickers: bool = False, only_pending: bool = False, only_live: bool = False, include_stickers: bool = False):
    """
    Load all shots with 12 ball channels, optionally + 8 sticker channels (20 total).
    The diameter filter is applied (detection quality), but NOT the distance filter.

    Returns:
        X:         (n, 12 or 20, SERIES_LENGTH)
        y:         (n,)
        dist_mask: (n,) bool — True if the shot passes the DIST_THRESHOLD filter
    """
    shots = []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "skip":
                continue
            if only_8_stickers and row.get("has_8_stickers", "").lower() not in ("true", "1", "yes"):
                continue
            shots.append(row)

    X_list, y_list, dist_list = [], [], []
    for shot in shots:
        dataset_name = shot["dataset_name"]
        parts    = Path(shot["rel_shot_dir"]).parts
        batch_id = next((p for p in parts if p.startswith(("pending_", "live_"))), "unknown")
        if only_pending and not batch_id.startswith("pending_"):
            continue
        if only_live and not batch_id.startswith("live_"):
            continue
        norm_csv = NORMALIZED_DIR / f"{batch_id}_{dataset_name}.csv"
        if not norm_csv.exists():
            continue

        rows = []
        with open(norm_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append([float(row["xn"]), float(row["yn"]), float(row["dist_n"]), float(row.get("diameter_norm", 0))])
        if not rows:
            continue

        seq = np.array(rows)
        seq = seq[seq[:, 3] >= MIN_DIAMETER_NORM]
        if len(seq) == 0:
            continue

        passes_dist = bool((seq[:, 2] < DIST_THRESHOLD).any())

        t_old = np.linspace(0, 1, len(seq))
        t_new = np.linspace(0, 1, SERIES_LENGTH)
        seq   = np.stack([np.interp(t_new, t_old, seq[:, c]) for c in range(4)], axis=1)  # (L, 4)

        vel = np.gradient(seq, axis=0)
        acc = np.gradient(vel,  axis=0)
        seq = np.concatenate([seq, vel, acc], axis=1)  # (L, 12)

        if include_stickers:
            sticker_csv = STICKER_TRACKING_DIR / f"{batch_id}_{dataset_name}.csv"
            if not sticker_csv.exists():
                continue
            sticker_rows = []
            with open(sticker_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sticker_rows.append([1.0 if row[f"sticker_{i}"] in ("True", "1") else 0.0 for i in range(1, 9)])
            if not sticker_rows:
                continue
            s = np.array(sticker_rows)  # (N_frames, 8)
            t_old_s = np.linspace(0, 1, len(s))
            s = np.stack([np.interp(t_new, t_old_s, s[:, c]) for c in range(8)], axis=1)  # (L, 8)
            seq = np.concatenate([seq, s], axis=1)  # (L, 20)

        X_list.append(seq.T)  # (12 or 20, L)
        y_list.append(1 if shot["label"] == "goal" else 0)
        dist_list.append(passes_dist)

    return np.stack(X_list), np.array(y_list), np.array(dist_list)


def run_experiment(X_train, y_train, X_test, y_test, ch):
    Xtr = X_train[:, ch, :]
    Xte = X_test[:, ch, :]
    rocket = MiniRocketMultivariate(num_kernels=10_000, random_state=42)
    rocket.fit(Xtr)
    Xtr_t = rocket.transform(Xtr)
    Xte_t = rocket.transform(Xte)
    scaler = StandardScaler()
    Xtr_t = scaler.fit_transform(Xtr_t)
    Xte_t = scaler.transform(Xte_t)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(Xtr_t, y_train)
    y_pred = (clf.decision_function(Xte_t) >= 0).astype(int)
    return balanced_accuracy_score(y_test, y_pred)

def train_on_data(X, y, experiments, label=""):
    print(f"  {len(y)} shots  (goals={y.sum()}, misses={(y == 0).sum()})")
    print(f"  Running {N_SEEDS} seeds per experiment...\n")

    idx = np.arange(len(y))
    results = []
    for exp in experiments:
        accs = []
        for seed in SEEDS:
            train_idx, test_idx = train_test_split(idx, test_size=TEST_FRACTION, stratify=y, random_state=seed)
            acc = run_experiment(X[train_idx], y[train_idx], X[test_idx], y[test_idx], exp["ch"])
            accs.append(acc)
        mean, std = np.mean(accs), np.std(accs)
        results.append((exp["name"], len(exp["ch"]), mean, std))
        print(f"  {exp['name']}  ->  {mean:.4f} (+/- {std:.4f})")

    header = f"  Results: {label}" if label else "  Results:"
    print("\n" + "=" * 62)
    print(header)
    print(f"  {'Experiment':<33} {'Ch':>3}  {'Bal. Acc':>8}  {'Std':>7}")
    print("-" * 62)
    for name, n_ch, mean, std in sorted(results, key=lambda r: -r[2]):
        print(f"  {name:<33} {n_ch:>3}  {mean:.4f}    {std:.4f}")
    print("=" * 62)

def main():

    # print("Loading all data...")
    # X, y, dist_mask = load_all_channels()
    # train_on_data(X, y, EXPERIMENTS_ALL, label="all data (no dist filter)")
    # train_on_data(X[dist_mask], y[dist_mask], EXPERIMENTS_ALL, label="all data (dist-filtered)")

    # print("\nLoading only-pending data...")
    # Xp, yp, dist_maskp = load_all_channels(only_pending=True)
    # train_on_data(Xp, yp, EXPERIMENTS_ALL, label="only pending (no dist filter)")
    # train_on_data(Xp[dist_maskp], yp[dist_maskp], EXPERIMENTS_ALL, label="only pending (dist-filtered)")

    # print("\nLoading only-live data...")
    # Xl, yl, dist_maskl = load_all_channels(only_live=True)
    # train_on_data(Xl, yl, EXPERIMENTS_ALL, label="only live (no dist filter)")
    # train_on_data(Xl[dist_maskl], yl[dist_maskl], EXPERIMENTS_ALL, label="only live (dist-filtered)")

    print("\nLoading only-8-stickers data...")
    X8, y8, dist_mask8 = load_all_channels(only_8_stickers=True, include_stickers=True)
    train_on_data(X8, y8, EXPERIMENTS_ONLY_8_STICKERS, label="only 8 stickers (no dist filter)")
    train_on_data(X8[dist_mask8], y8[dist_mask8], EXPERIMENTS_ONLY_8_STICKERS, label="only 8 stickers (dist-filtered)")



if __name__ == "__main__":
    main()
