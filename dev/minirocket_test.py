"""
MiniRocket binary classifier: goal vs miss from normalized ball trajectories.

Loads per-shot normalized CSVs from data/ball_tracking_normalized/, filters to
shots where the ball came within 1.1 normalized distance of the hoop center,
then trains and evaluates a MiniRocket classifier via 5-fold stratified CV.

Input features per shot: xn, yn, dist_n (3 channels, resampled to 50 timesteps)
Labels: goal=1, everything else=0

Run from repo root:
    python dev/minirocket_test.py
"""

import argparse
import csv
import math
from pathlib import Path

import joblib

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REPO_ROOT, LABELS_CSV as CSV_PATH, NORMALIZED_DIR, DIST_THRESHOLD, SERIES_LENGTH, CHANNELS

N_FOLDS = 10


def load_normalized_csv(path: Path) -> np.ndarray | None:
    """
    Load a normalized tracking CSV and return an array of shape (n_frames, 3).
    Returns None if the file is empty or unreadable.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append([float(row["xn"]), float(row["yn"]), float(row["dist_n"])])
    if not rows:
        return None
    return np.array(rows)


def resample(sequence: np.ndarray, length: int) -> np.ndarray:
    """
    Resample a (n_frames, n_channels) array to (length, n_channels) via
    linear interpolation along the time axis.
    """
    n = len(sequence)
    old_times = np.linspace(0, 1, n)
    new_times = np.linspace(0, 1, length)
    resampled = np.stack(
        [np.interp(new_times, old_times, sequence[:, c]) for c in range(sequence.shape[1])],
        axis=1,
    )
    return resampled


def add_derivatives(sequence: np.ndarray) -> np.ndarray:
    """
    Append velocity and acceleration channels to a (length, n_channels) array.
    Uses np.gradient (central differences) so output length is unchanged.
    Returns (length, n_channels * 3).
    """
    velocity     = np.gradient(sequence, axis=0)
    acceleration = np.gradient(velocity, axis=0)
    return np.concatenate([sequence, velocity, acceleration], axis=1)


def load_dataset():
    """
    Read shot_labels.csv, load normalized CSVs, apply dist_n < DIST_THRESHOLD
    filter, resample to SERIES_LENGTH.

    Returns:
        X: np.ndarray of shape (n_shots, n_channels, SERIES_LENGTH)
        y: np.ndarray of shape (n_shots,) with 1=goal, 0=other
    """
    shots = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "skip":
                continue
            shots.append(row)

    X_list, y_list, meta_list = [], [], []
    missing_by_batch: dict[str, int] = {}

    for shot in shots:
        dataset_name = shot["dataset_name"]
        parts        = Path(shot["rel_shot_dir"]).parts
        batch_id     = next((p for p in parts if p.startswith(("pending_", "live_"))), "unknown")
        norm_csv     = NORMALIZED_DIR / f"{batch_id}_{dataset_name}.csv"

        if not norm_csv.exists():
            missing_by_batch[batch_id] = missing_by_batch.get(batch_id, 0) + 1
            continue

        sequence = load_normalized_csv(norm_csv)
        if sequence is None:
            missing_by_batch[batch_id] = missing_by_batch.get(batch_id, 0) + 1
            continue

        min_dist = sequence[:, 2].min()   # dist_n column
        if min_dist >= DIST_THRESHOLD:
            continue

        resampled = resample(sequence, SERIES_LENGTH)     # (SERIES_LENGTH, 3)
        resampled = add_derivatives(resampled)           # (SERIES_LENGTH, 9)
        X_list.append(resampled.T)                       # (9, SERIES_LENGTH)
        y_list.append(1 if shot["label"] == "goal" else 0)
        meta_list.append({"batch_id": batch_id, "dataset_name": dataset_name, "label": shot["label"]})

    if missing_by_batch:
        total_missing = sum(missing_by_batch.values())
        print(f"  Note: {total_missing} shots skipped (missing normalized CSV):")
        for batch, count in sorted(missing_by_batch.items()):
            print(f"    {batch}: {count} shots")
        print()

    X = np.stack(X_list)   # (n_shots, 3, SERIES_LENGTH)
    y = np.array(y_list)
    return X, y, meta_list


def main():
    ap = argparse.ArgumentParser(description="MiniRocket CV evaluation.")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="Predictions with confidence below this are marked 'not sure' (default: 0, disabled)")
    ap.add_argument("--random-test", action="store_true",
                    help="Use a random stratified 20%% split for the final test set instead of the hardcoded batch")
    ap.add_argument("--save-model", action="store_true",
                    help="Train on ALL data (no held-out split) and save model to models/")
    ap.add_argument("--only-live", action="store_true",
                    help="Only use shots from live_* batches")
    args = ap.parse_args()
    min_confidence = args.min_confidence

    print("Loading data...")
    X, y, meta = load_dataset()

    if args.only_live:
        live_mask = np.array([m["batch_id"].startswith("live_") for m in meta])
        X, y, meta = X[live_mask], y[live_mask], [m for m, keep in zip(meta, live_mask) if keep]
        print(f"  Filtered to live_ batches only.")

    n_goals  = y.sum()
    n_misses = (y == 0).sum()
    print(f"  Shots used:  {len(y)}  (goals={n_goals}, misses={n_misses})")

    # --- save-model: run CV for accuracy estimate, then train on all data ---
    if args.save_model:
        models_dir = REPO_ROOT / "raspberry"
        models_dir.mkdir(exist_ok=True)

        print(f"\nRunning {N_FOLDS}-fold CV to estimate accuracy...")
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        accs, bal_accs = [], []
        fold_bar = tqdm(enumerate(cv.split(X, y), start=1), total=N_FOLDS, desc="CV folds", unit="fold")
        for fold, (train_idx, test_idx) in fold_bar:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            fold_bar.set_postfix_str("fitting MiniRocket")
            r = MiniRocketMultivariate(num_kernels=10_000, random_state=42)
            r.fit(X_tr)
            fold_bar.set_postfix_str("transforming")
            X_tr_t = r.transform(X_tr)
            X_te_t = r.transform(X_te)
            fold_bar.set_postfix_str("fitting classifier")
            s = StandardScaler()
            X_tr_t = s.fit_transform(X_tr_t)
            X_te_t = s.transform(X_te_t)
            c = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            c.fit(X_tr_t, y_tr)
            y_pred = (c.decision_function(X_te_t) >= 0).astype(int)
            acc     = accuracy_score(y_te, y_pred)
            bal_acc = balanced_accuracy_score(y_te, y_pred)
            accs.append(acc)
            bal_accs.append(bal_acc)
            fold_bar.set_postfix_str(f"acc={acc:.3f}  bal={bal_acc:.3f}")
            tqdm.write(f"  Fold {fold:2d}:  acc={acc:.3f}  balanced_acc={bal_acc:.3f}")

        print()
        print(f"  Mean accuracy (weighted):    {np.mean(accs):.3f}  (+/- {np.std(accs):.3f})")
        print(f"  Mean balanced accuracy:      {np.mean(bal_accs):.3f}  (+/- {np.std(bal_accs):.3f})")
        print()

        print("Training on all data and saving model...")
        rocket = MiniRocketMultivariate(num_kernels=10_000, random_state=42)
        rocket.fit(X)
        X_t = rocket.transform(X)

        scaler = StandardScaler()
        X_t = scaler.fit_transform(X_t)

        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        clf.fit(X_t, y)

        model = {
            "rocket":         rocket,
            "scaler":         scaler,
            "clf":            clf,
            "series_length":  SERIES_LENGTH,
            "channels":       CHANNELS,
            "dist_threshold": DIST_THRESHOLD,
        }
        out_path = models_dir / "minirocket_model.joblib"
        joblib.dump(model, out_path)
        print(f"  Written: {out_path}")
        return

    # --- held-out test split ---
    if args.random_test:
        all_idx = np.arange(len(y))
        train_idx_all, test_idx_final = train_test_split(
            all_idx, test_size=0.2, stratify=y
        )
        test_desc = "random stratified 20%"
    else:
        FINAL_TEST_BATCH = "pending_5014816"
        train_idx_all = np.array([i for i, m in enumerate(meta) if m["batch_id"] != FINAL_TEST_BATCH])
        test_idx_final = np.array([i for i, m in enumerate(meta) if m["batch_id"] == FINAL_TEST_BATCH])
        test_desc = f"batch={FINAL_TEST_BATCH}"

    X_train_all, X_final_test = X[train_idx_all], X[test_idx_final]
    y_train_all, y_final_test = y[train_idx_all], y[test_idx_final]
    meta_train      = [meta[i] for i in train_idx_all]
    meta_final_test = [meta[i] for i in test_idx_final]

    print(f"  Train+CV:    {len(y_train_all)}  shots")
    print(f"  Final test:  {len(y_final_test)}  shots  ({test_desc}, never seen during CV)\n")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    accs, bal_accs = [], []
    cm_total        = np.zeros((2, 2), dtype=int)
    wrong_indices   = []   # (orig_idx, score) for misclassified shots
    all_predictions = []   # (orig_idx, true, pred, score) for every shot

    fold_bar = tqdm(enumerate(cv.split(X_train_all, y_train_all), start=1), total=N_FOLDS, desc="CV folds", unit="fold")

    for fold, (train_idx, test_idx) in fold_bar:
        X_train, X_test = X_train_all[train_idx], X_train_all[test_idx]
        y_train, y_test = y_train_all[train_idx], y_train_all[test_idx]

        fold_bar.set_postfix_str("fitting MiniRocket")
        rocket = MiniRocketMultivariate(num_kernels=10_000, random_state=42)
        rocket.fit(X_train)

        fold_bar.set_postfix_str("transforming")
        X_train_t = rocket.transform(X_train)
        X_test_t  = rocket.transform(X_test)

        fold_bar.set_postfix_str("fitting classifier")
        scaler = StandardScaler()
        X_train_t = scaler.fit_transform(X_train_t)
        X_test_t  = scaler.transform(X_test_t)

        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        clf.fit(X_train_t, y_train)
        scores = clf.decision_function(X_test_t)   # positive = goal, negative = miss
        y_pred = (scores >= 0).astype(int)

        acc     = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        cm      = confusion_matrix(y_test, y_pred, labels=[0, 1])

        accs.append(acc)
        bal_accs.append(bal_acc)
        cm_total += cm

        for orig_idx, true, pred, score in zip(test_idx, y_test, y_pred, scores):
            all_predictions.append((int(orig_idx), int(true), int(pred), float(score)))
            if true != pred:
                wrong_indices.append((int(orig_idx), float(score)))  # index into meta_train

        fold_bar.set_postfix_str(f"acc={acc:.3f}  bal_acc={bal_acc:.3f}")
        tqdm.write(f"  Fold {fold}:  acc={acc:.3f}  balanced_acc={bal_acc:.3f}")

    print()
    print(f"  Mean accuracy:          {np.mean(accs):.3f}  (+/- {np.std(accs):.3f})")
    print(f"  Mean balanced accuracy: {np.mean(bal_accs):.3f}  (+/- {np.std(bal_accs):.3f})")
    print()
    print("  Confusion matrix (summed over folds):")
    print(f"                Predicted miss  Predicted goal")
    print(f"  Actual miss        {cm_total[0, 0]:4d}            {cm_total[0, 1]:4d}")
    print(f"  Actual goal        {cm_total[1, 0]:4d}            {cm_total[1, 1]:4d}")

    # --- not-sure analysis (CV) ---
    total = len(all_predictions)
    not_sure       = [(idx, true, pred, score) for idx, true, pred, score in all_predictions if abs(score) < min_confidence]
    not_sure_goals = [p for p in not_sure if p[1] == 1]
    not_sure_miss  = [p for p in not_sure if p[1] == 0]

    print()
    print(f"  Not sure (confidence < {min_confidence}):  {len(not_sure)} / {total} shots  ({100 * len(not_sure) / total:.1f}%)")
    if min_confidence > 0:
        print(f"    Goals converted to not sure:  {len(not_sure_goals)}")
        print(f"    Misses converted to not sure: {len(not_sure_miss)}")

    # --- wrong predictions with confidence (CV) ---
    print(f"\n  Wrong predictions ({len(wrong_indices)}):")
    for idx, score in sorted(wrong_indices, key=lambda t: -abs(t[1])):
        m           = meta_train[idx]
        shot_number = m["dataset_name"].split("_shot")[-1] if "_shot" in m["dataset_name"] else m["dataset_name"]
        not_sure_flag = "  [not sure]" if abs(score) < min_confidence else ""
        print(f"    batch={m['batch_id']}  shot={shot_number}  true={m['label']:4s}  confidence={abs(score):.4f}{not_sure_flag}")

    out_path = REPO_ROOT / "data" / "minirocket_wrong_predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_id", "shot_number", "true_label", "decision_score", "confidence", "not_sure"])
        writer.writeheader()
        for idx, score in sorted(wrong_indices, key=lambda t: t[0]):
            m = meta_train[idx]
            shot_number = m["dataset_name"].split("_shot")[-1] if "_shot" in m["dataset_name"] else m["dataset_name"]
            writer.writerow({
                "batch_id":       m["batch_id"],
                "shot_number":    shot_number,
                "true_label":     m["label"],
                "decision_score": round(score, 4),
                "confidence":     round(abs(score), 4),
                "not_sure":       abs(score) < min_confidence,
            })

    print(f"\n  Written: {out_path}")

    # --- final held-out test evaluation ---
    print()
    print("=" * 60)
    print(f"  FINAL TEST SET EVALUATION ({test_desc}, never seen during CV)")
    print("=" * 60)

    final_rocket = MiniRocketMultivariate(num_kernels=10_000, random_state=42)
    print("  Fitting final MiniRocket on all train data...")
    final_rocket.fit(X_train_all)
    X_train_all_t  = final_rocket.transform(X_train_all)
    X_final_test_t = final_rocket.transform(X_final_test)

    final_scaler = StandardScaler()
    X_train_all_t  = final_scaler.fit_transform(X_train_all_t)
    X_final_test_t = final_scaler.transform(X_final_test_t)

    final_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    final_clf.fit(X_train_all_t, y_train_all)
    final_scores = final_clf.decision_function(X_final_test_t)
    y_final_pred  = (final_scores >= 0).astype(int)

    final_acc     = accuracy_score(y_final_test, y_final_pred)
    final_bal_acc = balanced_accuracy_score(y_final_test, y_final_pred)
    final_cm      = confusion_matrix(y_final_test, y_final_pred, labels=[0, 1])

    print(f"  Accuracy:          {final_acc:.3f}")
    print(f"  Balanced accuracy: {final_bal_acc:.3f}")
    print()
    print("  Confusion matrix:")
    print(f"                Predicted miss  Predicted goal")
    print(f"  Actual miss        {final_cm[0, 0]:4d}            {final_cm[0, 1]:4d}")
    print(f"  Actual goal        {final_cm[1, 0]:4d}            {final_cm[1, 1]:4d}")

    final_wrong = [(i, int(true), int(pred), float(score))
                   for i, (true, pred, score) in enumerate(zip(y_final_test, y_final_pred, final_scores))
                   if true != pred]
    print(f"\n  Wrong predictions ({len(final_wrong)}):")
    for i, true, pred, score in sorted(final_wrong, key=lambda t: -abs(t[3])):
        m = meta_final_test[i]
        shot_number = m["dataset_name"].split("_shot")[-1] if "_shot" in m["dataset_name"] else m["dataset_name"]
        not_sure_flag = "  [not sure]" if abs(score) < min_confidence else ""
        print(f"    batch={m['batch_id']}  shot={shot_number}  true={m['label']:4s}  confidence={abs(score):.4f}{not_sure_flag}")

    final_out_path = REPO_ROOT / "data" / "minirocket_wrong_predictions_final_test.csv"
    with open(final_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_id", "shot_number", "true_label", "decision_score", "confidence", "not_sure"])
        writer.writeheader()
        for i, true, pred, score in sorted(final_wrong, key=lambda t: t[0]):
            m = meta_final_test[i]
            shot_number = m["dataset_name"].split("_shot")[-1] if "_shot" in m["dataset_name"] else m["dataset_name"]
            writer.writerow({
                "batch_id":       m["batch_id"],
                "shot_number":    shot_number,
                "true_label":     m["label"],
                "decision_score": round(score, 4),
                "confidence":     round(abs(score), 4),
                "not_sure":       abs(score) < min_confidence,
            })
    print(f"\n  Written: {final_out_path}")


if __name__ == "__main__":
    main()
