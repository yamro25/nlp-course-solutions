import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import config
from src.data.preprocessor import clean_text, load_vocab, encode_and_pad
from src.data.dataset import JigsawDataset
from src.models.lstm_classifier import ToxicLSTM


def evaluate_with_best_thresholds(targets: np.ndarray, preds_prob: np.ndarray, label_names: list):
    """
    Finds the optimal probability threshold per class to maximize F1-Score,
    and prints a comprehensive evaluation table.
    """
    print("\n" + "=" * 82)
    print(f"{'Label':<16} | {'Best Thresh':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("=" * 82)

    for i, col in enumerate(label_names):
        best_th, best_f1, best_prec, best_rec = 0.5, 0.0, 0.0, 0.0
        
        # البحث عن أفضل عتبة توازن بين الـ Precision والـ Recall
        for th in np.arange(0.30, 0.96, 0.02):
            bin_p = (preds_prob[:, i] >= th).astype(int)
            f1 = f1_score(targets[:, i], bin_p, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                best_prec = precision_score(targets[:, i], bin_p, zero_division=0)
                best_rec = recall_score(targets[:, i], bin_p, zero_division=0)

        try:
            auc = roc_auc_score(targets[:, i], preds_prob[:, i])
        except ValueError:
            auc = 0.0

        print(f"{col:<16} | {best_th:<12.2f} | {best_prec:<10.4f} | {best_rec:<10.4f} | {best_f1:<10.4f} | {auc:<10.4f}")
    print("=" * 82)


def run_evaluation():
    print("[1/3] Loading Artifacts...")
    vocab = load_vocab(config.VOCAB_SAVE_PATH)
    df = pd.read_csv(config.TRAIN_DATA_PATH)
    
    # تنظيف النصوص وضمان عدم وجود قيم فارغة
    df['comment_text'] = df['comment_text'].fillna("")
    df['tokens'] = df['comment_text'].apply(clean_text)
    sequences = [encode_and_pad(tok, vocab, config.MAX_LEN) for tok in df['tokens']]
    labels = df[config.LABEL_COLS].values

    # تقسيم البيانات للحصول على نفس جزء الـ Test
    _, X_test, _, y_test = train_test_split(sequences, labels, test_size=0.15, random_state=42)
    test_loader = DataLoader(JigsawDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)

    print("[2/3] Loading Model...")
    model = ToxicLSTM(len(vocab), config.EMBED_DIM, config.HIDDEN_DIM, len(config.LABEL_COLS)).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    print("[3/3] Generating Predictions & Computing Optimal Thresholds...")
    all_preds = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(config.DEVICE)
            probs = torch.sigmoid(model(x_batch)).cpu().numpy()
            all_preds.append(probs)

    preds_prob = np.vstack(all_preds)
    evaluate_with_best_thresholds(y_test, preds_prob, config.LABEL_COLS)


if __name__ == "__main__":
    run_evaluation()