import gc
from typing import List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.optim import AdamW
from tqdm.auto import tqdm

from marble.train.scheduler import WarmupCosineScheduler


class ClassificationMetrics(NamedTuple):
    """Fixed-shape evaluate() return. Indexing [0]/[2] matches the prior tuple layout."""
    acc: float
    auc: float
    report: dict
    per_class_aucs: Optional[List[float]] = None


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, scaler=None, max_grad_norm=0.5):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False, miniters=50):
        x      = batch['mamba_input'].to(device, non_blocking=True)
        scales = batch['scale'].to(device, non_blocking=True)
        mask   = batch['mask'].to(device, non_blocking=True)
        y      = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast('cuda', enabled=False):
                logits, _ = model(x, scales, mask)
                loss   = criterion(logits, y)

            torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(x, scales, mask)
            loss   = criterion(logits, y)
            torch.cuda.empty_cache()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * y.size(0)

        del x, scales, mask, y, logits, loss
        gc.collect()

    if scheduler:
        scheduler.step()

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device) -> ClassificationMetrics:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            x      = batch['mamba_input'].to(device, non_blocking=True)
            scales = batch['scale'].to(device, non_blocking=True)
            mask   = batch['mask'].to(device, non_blocking=True)
            y      = batch['label'].to(device, non_blocking=True)

            logits, _ = model(x, scales, mask)    # (B, C)
            probs  = F.softmax(logits, dim=1)  # (B, C)

            # replace any NaN in probs with uniform probability
            # (1/C gives neutral signal for AUC)
            probs = torch.nan_to_num(probs, nan=1.0/probs.size(1))

            preds = probs.argmax(dim=1)        # (B,)

            all_labels.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            # for binary we log only the positive class; multiclass we log all
            if probs.size(1) == 2:
                all_probs.append(probs[:,1].cpu().numpy())
            else:
                all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds,  axis=0)
    y_prob = np.concatenate(all_probs,  axis=0)

    # accuracy
    acc = accuracy_score(y_true, y_pred)

    # ROC-AUC (binary or macro multiclass)
    per_class_aucs = None
    if y_prob.ndim == 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        # multiclass: compute one-vs-rest and average
        per_class_aucs = []
        for c in range(y_prob.shape[1]):
            per_class_aucs.append(roc_auc_score((y_true==c).astype(int), y_prob[:,c]))
        auc = float(np.mean(per_class_aucs))

    # Per class metrics
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    return ClassificationMetrics(acc=acc, auc=auc, report=report, per_class_aucs=per_class_aucs)


def run_training(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        base_lr=3e-5,
        weight_decay=1e-2,
        epochs=20,
        warmup_epochs=2,
        use_amp=True,
        patience=3,
        model_path=None,
        train_weights=None,
        train_sampler=None
):
    assert model_path is not None, "Model path must be provided for loading the model."
    # Setup
    weights = torch.tensor(train_weights, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    total_steps = epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_epochs, total_steps=total_steps)
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_auc = 0.0
    epochs_no_improve = 0
    best_val_metrics = None
    best_epoch = 0

    for epoch in range(1, epochs+1):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, scaler=scaler, max_grad_norm=1.0
        )
        val_metrics = evaluate(model, val_loader, device)
        val_acc, val_auc = val_metrics.acc, val_metrics.auc

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # Early stopping on validation AUC after warmup
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_metrics = val_metrics
            best_epoch = epoch
            epochs_no_improve = 0
            # unwrap DataParallel if present
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), model_path)
            print("  ↳ New best AUC, saving model.")
        else:
            if epoch > warmup_epochs:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"→ Early stopping at epoch {epoch}")
                    break

    state_dict = torch.load(model_path, map_location=device)
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    print("Loaded best model for final evaluation")

    test_metrics = evaluate(model, test_loader, device)
    test_acc, test_auc = test_metrics.acc, test_metrics.auc
    print(f"Test Accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

    return best_val_metrics, test_metrics, best_epoch
