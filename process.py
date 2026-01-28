"""Training / validation utilities for HyperPep.

- train(): uses BCEWithLogitsLoss with a *global* pos_weight to address class imbalance.
- validation(): uses unweighted BCE for reporting, and returns ROC/PR curves.

Both functions accumulate predictions across the whole dataloader to compute metrics.
"""

# process.py —— 使用“全局 pos_weight”训练；验证仍用无权重 BCE
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef
)

def train(model, device, dataloader, optim, epoch, pos_weight_global: float):
    """One training epoch.

    Args:
      pos_weight_global: scalar weight for positive class in BCEWithLogitsLoss.
        Typically = (N_neg / N_pos). Keep it fixed to avoid per-batch jitter.
    """
    model.train()
    loss_collect = 0.0

    # 固定的全局 pos_weight（标量）
    # Fixed scalar pos_weight to counter class imbalance (kept constant across batches).
    posw = torch.tensor(float(pos_weight_global), device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=posw, reduction='mean')

    TP = TN = FP = FN = 0
    y_true_all, y_pred_all = [], []

    for batch in dataloader:
        # Each batch is a PyG Batch object containing concatenated samples.
        # Move it to GPU/CPU once, then run forward/backward.
        optim.zero_grad()
        batch = batch.to(device)

        # Model returns logits (before sigmoid).
        z = model(
            x_h=batch.x_h,
            h2_edge_index=batch.h2_edge_index,
            h2_edge_attr=batch.h2_edge_attr,
            idx_batch=batch.batch,
            num_hyper2edges=batch.num_hyper2edges  # 保持传入，避免非确定性
        )

        # Ground-truth labels as float tensor (shape [B]).
        y = batch.y.float().view(-1)
        loss = bce(z, y)

        loss.backward()
        optim.step()
        loss_collect += loss.item()

        # Convert logits to probabilities for threshold-based statistics.
        probs = torch.sigmoid(z).view(-1)
        preds = (probs > 0.5).long()
        labels = y.long()
        y_true_all.extend(labels.cpu().numpy())
        y_pred_all.extend(preds.cpu().numpy())

        TP += ((preds == 1) & (labels == 1)).sum().item()
        TN += ((preds == 0) & (labels == 0)).sum().item()
        FP += ((preds == 1) & (labels == 0)).sum().item()
        FN += ((preds == 0) & (labels == 1)).sum().item()

    # loss_collect is summed over batches; dividing by dataset size approximates per-sample loss.
    loss_collect /= max(len(dataloader.dataset), 1)
    ACC = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    SN  = TP / (TP + FN + 1e-8)
    SP  = TN / (TN + FP + 1e-8)
    MCC = matthews_corrcoef(y_true_all, y_pred_all) if len(set(y_true_all)) > 1 else 0.0

    print(f"Epoch: {epoch}  Loss/Data: {loss_collect * 100:.4f}%  "
          f"ACC={ACC:.4f} SN={SN:.4f} SP={SP:.4f} MCC={MCC:.4f}")
    print('---------------------------------------')
    return loss_collect

def validation(model, device, dataloader, epoch):
    """Evaluate a model on a dataloader.

    Returns:
      loss_collect, AUC, AUPR, fpr, tpr, precision, recall, y_true_all, y_prob_all
    """
    model.eval()
    loss_collect = 0.0

    TP = TN = FP = FN = 0
    y_true_all, y_pred_all, y_prob_all = [], [], []
    # Validation loss is unweighted BCE (for comparable reporting).
    bce_val = torch.nn.BCEWithLogitsLoss()  # 验证仍用“无权重 BCE”

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Forward pass (returns logits).
            z = model(
                x_h=batch.x_h,
                h2_edge_index=batch.h2_edge_index,
                h2_edge_attr=batch.h2_edge_attr,
                idx_batch=batch.batch,
                num_hyper2edges=batch.num_hyper2edges
            )
            y = batch.y.float().view(-1)

            loss = bce_val(z, y)
            loss_collect += loss.item()

            # logits -> probabilities
            probs = torch.sigmoid(z).view(-1)
            preds = (probs > 0.5).long()
            labels = y.long()

            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
            y_prob_all.extend(probs.cpu().numpy())

            TP += ((preds == 1) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

    loss_collect /= max(len(dataloader.dataset), 1)
    ACC = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    SN = TP / (TP + FN + 1e-8)
    SP = TN / (TN + FP + 1e-8)
    MCC = matthews_corrcoef(y_true_all, y_pred_all) if len(set(y_true_all)) > 1 else 0.0
    # AUC / AUPR require both classes to be present in y_true_all.
    try:
        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
        AUC = roc_auc_score(y_true_all, y_prob_all)
        precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
        AUPR = average_precision_score(y_true_all, y_prob_all)
    except ValueError:
        AUC = float('nan'); AUPR = float('nan')
        fpr = tpr = precision = recall = None


    print(f"ACC={ACC:.4f}   SN={SN:.4f}   SP={SP:.4f}   MCC={MCC:.4f}   AUC={AUC:.4f}   AUPR={AUPR:.4f}")
    print('---------------------------------------')

    return loss_collect, AUC, AUPR, fpr, tpr, precision, recall, y_true_all, y_prob_all
