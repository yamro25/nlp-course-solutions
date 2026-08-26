import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from src.data.preprocessor import clean_text, build_vocab, encode_and_pad, save_vocab
from src.data.dataset import JigsawDataset
from src.models.lstm_classifier import ToxicLSTM
from src.utils.metrics import calculate_pos_weights

def run_training():
    os.makedirs(os.path.join(config.BASE_DIR,"saved_models"), exist_ok=True)
    os.makedirs(os.path.join(config.BASE_DIR,"data/processed"), exist_ok=True)

    print("[1/4] Loading and Preprocessing Data...")
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    df['comment_text'] = df['comment_text'].fillna("")
    df = df[df['comment_text'].str.strip() != ""]

    df['tokens'] = df['comment_text'].apply(clean_text)
    
    vocab = build_vocab(df['tokens'], config.MAX_VOCAB_SIZE)
    save_vocab(vocab, config.VOCAB_SAVE_PATH)
    print(f"Vocabulary Size: {len(vocab)}")

    sequences = [encode_and_pad(tok, vocab, config.MAX_LEN) for tok in df['tokens']]
    labels = df[config.LABEL_COLS].values

    # Train / Val Split (85% train, 15% validation)
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.15, random_state=42)

    train_loader = DataLoader(JigsawDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(JigsawDataset(X_val, y_val), batch_size=config.BATCH_SIZE)

    print("[2/4] Initializing Model & Loss...")
    model = ToxicLSTM(len(vocab), config.EMBED_DIM, config.HIDDEN_DIM, len(config.LABEL_COLS)).to(config.DEVICE)
    pos_weights = calculate_pos_weights(y_train, config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("[3/4] Starting Training Loop...")
    best_val_loss = float('inf')

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
                logits = model(x_batch)
                val_loss += criterion(logits, y_batch).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"--> Saved Best Model to {config.MODEL_SAVE_PATH}")

    print("[4/4] Training Complete.")

if __name__ == "__main__":
    run_training()