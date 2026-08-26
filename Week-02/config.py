import torch
import os


# Base directory (Root folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/raw/train.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models/best_lstm_model.pt")
VOCAB_SAVE_PATH = os.path.join(BASE_DIR, "data/processed/vocab.pkl")

# Labels
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Model & Training Hyperparameters
# إعدادات مخففة للـ CPU
MAX_VOCAB_SIZE = 25000   # بدل 25000
MAX_LEN = 80             # بدل 150 (يقلل وقت معالجة الـ LSTM للنصف)
EMBED_DIM = 64           # بدل 128
HIDDEN_DIM = 64          # بدل 128
BATCH_SIZE = 128         # حجم دفعة أكبر لتسريع الـ CPU
EPOCHS = 4               # بدل 4
LEARNING_RATE = 1e-3
THRESHOLD = 0.5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")