"""
vision_test.py

Benchmark a single-frame MobileNetV3-Small classifier for Stage-2 basketball shots.
Mirrors the structure of dev/minirocket_test.py for direct comparison.

For each Stage-2 shot (dist_n < 1.1):
- Selects the frame where dist_n is minimal (ball closest to hoop)
- Crops a fixed square around the hoop center using the per-shot ellipse
- Resizes to 224x224 and classifies with pretrained MobileNetV3-Small
  (backbone frozen, only the final linear layer trained)

Evaluation:
- GroupKFold cross-validation grouped by recording batch
- Same final held-out test batch as minirocket_test.py (pending_5014816)
- Primary metric: balanced accuracy

Run from repo root:
    python dev/vision_test.py
    python dev/vision_test.py --epochs 50 --margin 150

Requirements:
    pip install torch torchvision pillow
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REPO_ROOT, LABELS_CSV as CSV_PATH, NORMALIZED_DIR, GLOBAL_ELLIPSE, DIST_THRESHOLD
FINAL_TEST_BATCH = "pending_5014816"
N_FOLDS          = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class ShotRecord:
    batch_id: str
    dataset_name: str
    label: int                        # 1=goal, 0=miss
    label_str: str
    frame_path: Path
    ellipse_center: Tuple[float, float]


def _load_ellipse_center(ellipse_meta_rel: str) -> Tuple[float, float]:
    path = (REPO_ROOT / ellipse_meta_rel) if ellipse_meta_rel else None
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(GLOBAL_ELLIPSE.read_text(encoding="utf-8"))
    e = data.get("ellipse", data)
    return float(e["center"][0]), float(e["center"][1])


def _find_argmin_frame(batch_id: str, dataset_name: str) -> Optional[int]:
    """Return frame_index with lowest dist_n. Returns None if min dist >= threshold."""
    norm_csv = NORMALIZED_DIR / f"{batch_id}_{dataset_name}.csv"
    if not norm_csv.exists():
        return None
    best_idx, best_dist = None, float("inf")
    with open(norm_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = float(row["dist_n"])
            if d < best_dist:
                best_dist = d
                best_idx = int(row["frame_index"])
    return best_idx if best_dist < DIST_THRESHOLD else None


def _find_frame_file(shot_dir: Path, dataset_name: str, frame_index: int) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = shot_dir / f"{dataset_name}_{frame_index:06d}{ext}"
        if p.exists():
            return p
    return None


def load_records() -> List[ShotRecord]:
    records, skipped = [], 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] not in {"goal", "miss"}:
                continue

            parts    = Path(row["rel_shot_dir"]).parts
            batch_id = next((p for p in parts if p.startswith(("pending_", "live_"))), None)
            if not batch_id:
                skipped += 1
                continue

            dataset_name = row["dataset_name"]
            frame_idx    = _find_argmin_frame(batch_id, dataset_name)
            if frame_idx is None:
                skipped += 1
                continue

            frame_path = _find_frame_file(REPO_ROOT / row["rel_shot_dir"], dataset_name, frame_idx)
            if frame_path is None:
                skipped += 1
                continue

            try:
                center = _load_ellipse_center(row.get("ellipse_meta", ""))
            except Exception:
                skipped += 1
                continue

            records.append(ShotRecord(
                batch_id=batch_id,
                dataset_name=dataset_name,
                label=1 if row["label"] == "goal" else 0,
                label_str=row["label"],
                frame_path=frame_path,
                ellipse_center=center,
            ))

    if skipped:
        print(f"  Note: {skipped} shots skipped (missing frame, tracking CSV, or ellipse)")
    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _make_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class HoopCropDataset(Dataset):
    def __init__(self, records: List[ShotRecord], margin: int, train: bool):
        self.records   = records
        self.margin    = margin
        self.transform = _make_transforms(train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r  = self.records[idx]
        img = Image.open(r.frame_path).convert("RGB")
        w, h = img.size
        cx, cy = r.ellipse_center
        x0 = max(0, int(cx) - self.margin)
        y0 = max(0, int(cy) - self.margin)
        x1 = min(w, int(cx) + self.margin)
        y1 = min(h, int(cy) + self.margin)
        return self.transform(img.crop((x0, y0, x1, y1))), r.label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    # Replace only the final linear layer; rest of classifier stays pretrained/frozen
    model.classifier[3] = nn.Linear(1024, 2)
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device) -> None:
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        criterion(model(imgs), labels).backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (true_labels, predicted_labels, goal_probabilities)."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_true.extend(labels.numpy())
            all_pred.extend(preds)
            all_prob.extend(probs)
    return np.array(all_true), np.array(all_pred), np.array(all_prob)


def _print_wrong(wrong: list) -> None:
    """Print wrong predictions sorted by confidence (most confident wrong first)."""
    for r, true, pred, prob in sorted(wrong, key=lambda t: -abs(t[3] - 0.5)):
        shot_number = r.dataset_name.split("_shot")[-1] if "_shot" in r.dataset_name else r.dataset_name
        confidence  = abs(prob - 0.5) * 2
        print(f"    batch={r.batch_id}  shot={shot_number}  true={r.label_str:4s}  goal_prob={prob:.4f}  confidence={confidence:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Single-frame MobileNetV3-Small benchmark.")
    ap.add_argument("--margin",     type=int,   default=130, help="Pixel half-size of hoop crop (default: 130)")
    ap.add_argument("--epochs",     type=int,   default=30,  help="Training epochs per fold (default: 30)")
    ap.add_argument("--batch-size", type=int,   default=16,  help="DataLoader batch size (default: 16)")
    ap.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    args = ap.parse_args()

    device = torch.device("cpu")

    print("Loading data...")
    records  = load_records()
    n_goals  = sum(r.label for r in records)
    n_misses = len(records) - n_goals
    print(f"  Shots used:  {len(records)}  (goals={n_goals}, misses={n_misses})")

    train_records = [r for r in records if r.batch_id != FINAL_TEST_BATCH]
    test_records  = [r for r in records if r.batch_id == FINAL_TEST_BATCH]
    print(f"  Train+CV:    {len(train_records)}  shots")
    print(f"  Final test:  {len(test_records)}  shots  (batch={FINAL_TEST_BATCH}, never seen during CV)\n")

    # Inverse-frequency class weights for loss
    n_tr_goals  = sum(r.label for r in train_records)
    n_tr_misses = len(train_records) - n_tr_goals
    class_weights = torch.tensor([
        len(train_records) / (2.0 * n_tr_misses),
        len(train_records) / (2.0 * n_tr_goals),
    ])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    groups = [r.batch_id for r in train_records]
    y_tr   = np.array([r.label for r in train_records])
    kf     = GroupKFold(n_splits=N_FOLDS)

    bal_accs = []
    cm_total = np.zeros((2, 2), dtype=int)
    wrong_cv: list = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_records, y_tr, groups=groups), start=1):
        fold_train = [train_records[i] for i in tr_idx]
        fold_val   = [train_records[i] for i in val_idx]

        train_loader = DataLoader(HoopCropDataset(fold_train, args.margin, train=True),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(HoopCropDataset(fold_val,   args.margin, train=False),
                                  batch_size=args.batch_size, shuffle=False)

        model     = build_model().to(device)
        optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr=args.lr)

        for _ in range(args.epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)

        true, pred, prob = evaluate(model, val_loader, device)
        bal_acc = balanced_accuracy_score(true, pred)
        cm      = confusion_matrix(true, pred, labels=[0, 1])

        bal_accs.append(bal_acc)
        cm_total += cm

        for i, val_i in enumerate(val_idx):
            if true[i] != pred[i]:
                wrong_cv.append((train_records[val_i], int(true[i]), int(pred[i]), float(prob[i])))

        val_batches = len(set(r.batch_id for r in fold_val))
        print(f"  Fold {fold:2d}/{N_FOLDS}:  bal_acc={bal_acc:.3f}  (val: {len(fold_val)} shots, {val_batches} batch(es))")

    print()
    print(f"  Mean balanced accuracy: {np.mean(bal_accs):.3f}  (+/- {np.std(bal_accs):.3f})")
    print()
    print("  Confusion matrix (summed over folds):")
    print(f"                Predicted miss  Predicted goal")
    print(f"  Actual miss        {cm_total[0, 0]:4d}            {cm_total[0, 1]:4d}")
    print(f"  Actual goal        {cm_total[1, 0]:4d}            {cm_total[1, 1]:4d}")
    print(f"\n  Wrong predictions CV ({len(wrong_cv)}):")
    _print_wrong(wrong_cv)

    # --- Final held-out test ---
    print()
    print("=" * 60)
    print(f"  FINAL TEST SET EVALUATION (batch={FINAL_TEST_BATCH}, never seen during CV)")
    print("=" * 60)

    if not test_records:
        print("  No test shots found for this batch.")
        return

    train_loader_full = DataLoader(HoopCropDataset(train_records, args.margin, train=True),
                                   batch_size=args.batch_size, shuffle=True)
    test_loader       = DataLoader(HoopCropDataset(test_records,  args.margin, train=False),
                                   batch_size=args.batch_size, shuffle=False)

    print("  Training final model on all train data...")
    final_model     = build_model().to(device)
    final_optimizer = torch.optim.Adam(final_model.classifier[3].parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_epoch(final_model, train_loader_full, final_optimizer, criterion, device)
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1}/{args.epochs}")

    true, pred, prob = evaluate(final_model, test_loader, device)
    final_bal_acc = balanced_accuracy_score(true, pred)
    final_cm      = confusion_matrix(true, pred, labels=[0, 1])

    print(f"  Balanced accuracy: {final_bal_acc:.3f}")
    print()
    print("  Confusion matrix:")
    print(f"                Predicted miss  Predicted goal")
    print(f"  Actual miss        {final_cm[0, 0]:4d}            {final_cm[0, 1]:4d}")
    print(f"  Actual goal        {final_cm[1, 0]:4d}            {final_cm[1, 1]:4d}")

    wrong_final = [(test_records[i], int(true[i]), int(pred[i]), float(prob[i]))
                   for i in range(len(true)) if true[i] != pred[i]]
    print(f"\n  Wrong predictions ({len(wrong_final)}):")
    _print_wrong(wrong_final)


if __name__ == "__main__":
    main()
