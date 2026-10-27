import gc
import sys

import numpy as np
import torch
import torch.optim as optim
from sksurv.metrics import concordance_index_censored
from tqdm.auto import tqdm

from marble.train.losses import NLLSurvLoss
from marble.train.scheduler import WarmupCosineScheduler


def train_one_survival_epoch(model, loader, optimizer, criterion, device, scheduler=None, scaler=None, max_grad_norm=0.5):
    model.train()
    train_loss_surv, train_loss = 0., 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, batch in tqdm(enumerate(loader), desc="Training", miniters=50, file=sys.stdout, leave=False, total=len(loader)):
        x      = batch['mamba_input'].to(device, non_blocking=True)
        scales = batch['scale'].to(device, non_blocking=True)
        mask   = batch['mask'].to(device, non_blocking=True)
        label      = batch['label'].to(device, non_blocking=True)
        event_time = batch['event_time'].to(device, non_blocking=True)
        censorship = batch['censorship'].to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast('cuda', enabled=False):
                hazards, S = model(x, scales, mask)
                loss = criterion(hazards=hazards, S=S, Y=label, c=censorship)
            torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            hazards, S = model(x, scales, mask)
            loss = criterion(hazards=hazards, S=S, Y=label, c=censorship)
            torch.cuda.empty_cache()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        loss_value = loss.item()

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censorship.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value

        del x, scales, mask, label, event_time, censorship, hazards, S, loss
        gc.collect()

    if scheduler:
        scheduler.step()

    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    return train_loss_surv, train_loss, c_index


def evaluate_survival(model, loader, criterion, device):
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x      = batch['mamba_input'].to(device, non_blocking=True)
            scales = batch['scale'].to(device, non_blocking=True)
            mask   = batch['mask'].to(device, non_blocking=True)
            y      = batch['label'].to(device, non_blocking=True)
            event_time = batch['event_time'].to(device, non_blocking=True)
            censorship = batch['censorship'].to(device, non_blocking=True)

            hazards, S = model(x, scales, mask)    # (B, C)

            loss = criterion(hazards=hazards, S=S, Y=y, c=censorship, alpha=0)
            loss_value = loss.item()

            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = censorship.cpu().numpy()
            all_event_times[batch_idx] = event_time

            val_loss_surv += loss_value
            val_loss += loss_value

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    return val_loss_surv, val_loss, c_index


def run_training_survival(
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

    loss_fn = NLLSurvLoss(alpha=0)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, weight_decay=weight_decay)

    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_epochs, total_steps=epochs)
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None

    max_val_c_index = 0
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, epochs+1):
        train_sampler.set_epoch(epoch)
        train_loss_surv, train_loss, train_c_index = train_one_survival_epoch(
            model, train_loader, optimizer, loss_fn, device,
            scheduler=scheduler, scaler=scaler, max_grad_norm=1.0
        )
        val_loss_surv, val_loss, val_c_index = evaluate_survival(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val C-Index: {val_c_index:.4f}")

        # Early stopping on validation C-Index after warmup
        if val_c_index > max_val_c_index:
            max_val_c_index = val_c_index
            best_epoch = epoch
            epochs_no_improve = 0
            # unwrap DataParallel if present
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), model_path)
            print("  ↳ New best C-Index, saving model.")
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

    return max_val_c_index, best_epoch
