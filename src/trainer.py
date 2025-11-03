import time

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from IPython.display import display



@torch.no_grad()
def compute_accuracy(model, data_loader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0

    for batch in data_loader:
        labels = batch.pop('label').to(device, non_blocking=True)
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        
        logits = model(**inputs)
        preds = torch.argmax(logits, dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return correct / total

def train_model(model, train_loader, val_loader, test_loader,
                optimizer, criterion, scheduler, device, epochs, patience):
    history_rows = []
    model.to(device)

    best_val_acc, best_val_epoch = -1.0, -1
    best_test_acc, best_test_epoch = -1.0, -1
    best_induced_test_acc, best_induced_test_epoch = -1.0, -1
    best_induced_test_state = None
    
    epochs_no_improve = 0
    actual_epochs_run = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_correct, n_samples = 0.0, 0, 0
        t0 = time.time()

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for batch in loop:
            labels = batch.pop('label').to(device, non_blocking=True)
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            n_samples += bs
            epoch_loss += loss.item() * bs
            epoch_correct += (logits.argmax(dim=1) == labels).sum().item()

            loop.set_postfix({
                'loss': epoch_loss / n_samples,
                'acc': epoch_correct / n_samples
            })

        train_loss = epoch_loss / max(n_samples, 1)
        train_acc  = epoch_correct / max(n_samples, 1)

        val_acc  = compute_accuracy(model, val_loader, device)
        test_acc = compute_accuracy(model, test_loader, device)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc, best_test_epoch = test_acc, epoch

        if val_acc > best_val_acc:
            best_val_acc, best_val_epoch = val_acc, epoch
            best_induced_test_acc, best_induced_test_epoch = test_acc, epoch
            best_induced_test_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        history_rows.append({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Acc":  train_acc,
            "Val Acc":    val_acc,
            "Test Acc":   test_acc,
            "Time (s)":   round(time.time() - t0, 2),
        })
        if patience is not None and epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch} (no improvement for {patience} epochs).")
            actual_epochs_run = epoch
            break

    history = pd.DataFrame(history_rows)

    summary = {
        "epochs": actual_epochs_run,
        "total_train_time": history["Time (s)"].sum(),
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "best_test_acc": best_test_acc,
        "best_test_epoch": best_test_epoch,
        "best_induced_test_acc": best_induced_test_acc,
        "best_induced_test_epoch": best_induced_test_epoch,
        "history_df": history,
        "best_state_dict": best_induced_test_state
    }

    return summary

def print_summary(summary):
    
    history_df = summary.get('history_df')
    print("--- Epoch History ---")
    history_display = history_df.copy()
    history_df = history_df.set_index('Epoch')
    history_df = history_display.style.format({
        'Train Loss': '{:.4f}',
        'Train Acc': '{:.2%}',
        'Val Acc': '{:.2%}',
        'Test Acc': '{:.2%}',
        'Time (s)': '{:.2f}'
    })

    display(history_df)
    
    print("--- Training Summary ---")
    print(f"Total epochs run: {summary['epochs']}")
    print(f"Total training time: {summary['total_train_time']:.2f} seconds\n")
    
    print(f"Best Validation Accuracy: {summary['best_val_acc'] * 100:.2f}% (at Epoch {summary['best_val_epoch']})")
    print(f"Induced Test Accuracy: {summary['best_induced_test_acc'] * 100:.2f}% (at Epoch {summary['best_induced_test_epoch']})")
    print(f"Best Ever Test Accuracy: {summary['best_test_acc'] * 100:.2f}% (at Epoch {summary['best_test_epoch']})\n")