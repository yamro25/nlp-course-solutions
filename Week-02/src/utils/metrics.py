import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def calculate_pos_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Calculates class weights to handle severe label imbalance."""
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts
    weights = neg_counts / np.maximum(pos_counts, 1)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def evaluate_predictions(targets: np.ndarray, preds_prob: np.ndarray, label_names: list, threshold: float = 0.5):
    """Prints evaluation metrics per label."""
    binary_preds = (preds_prob >= threshold).astype(int)
    print("\n" + "=" * 65)
    print(f"{'Label':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("=" * 65)
    
    for i, col in enumerate(label_names):
        prec = precision_score(targets[:, i], binary_preds[:, i], zero_division=0)
        rec = recall_score(targets[:, i], binary_preds[:, i], zero_division=0)
        f1 = f1_score(targets[:, i], binary_preds[:, i], zero_division=0)
        try:
            auc = roc_auc_score(targets[:, i], preds_prob[:, i])
        except ValueError:
            auc = 0.0
        print(f"{col:<16} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {auc:<10.4f}")
    print("=" * 65)