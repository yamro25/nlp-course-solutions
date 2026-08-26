# Toxic Comment Classification (Jigsaw Challenge)

A modular multi-label text classification pipeline built with **PyTorch** using an **Embedding + Bidirectional LSTM** architecture to identify toxicity across 6 distinct categories in Wikipedia comments.

---

## 📌 Project Overview

* **Task:** Multi-Label Text Classification
* **Dataset:** Jigsaw Toxic Comment Classification Challenge (159,571 comments)
* **Labels:** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
* **Loss Function:** `nn.BCEWithLogitsLoss` with positive class weighting
* **Architecture:** Text Cleaner $\to$ Word Vocabulary $\to$ Embedding $\to$ Bi-LSTM $\to$ Max Pooling $\to$ Dense Classification Layer

---

## 📁 Directory Structure

```text
Week-02/
├── data/
│   ├── raw/                   # Place train.csv here
│   └── processed/             # Auto-generated vocab.pkl
├── src/
│   ├── data/
│   │   ├── dataset.py         # PyTorch Dataset implementation
│   │   └── preprocessor.py    # Cleaning, Tokenization, Vocabulary, Padding
│   ├── models/
│   │   └── lstm_classifier.py # ToxicLSTM PyTorch Architecture
│   └── utils/
│       └── metrics.py         # Multi-label evaluation (Precision, Recall, F1, AUC)
├── saved_models/              # Auto-saved best model checkpoint (best_lstm_model.pt)
├── config.py                  # Hyperparameters and paths
├── train.py                   # Training pipeline script
├── evaluate.py                # Model evaluation and metrics reporting script
└── README.md



# Results 

train.py
[1/4] Loading and Preprocessing Data...
Vocabulary Size: 25000
[2/4] Initializing Model & Loss...
[3/4] Starting Training Loop...
Epoch [1/4] | Train Loss: 0.7114 | Val Loss: 0.4650
--> Saved Best Model to d:\AI\OmarNlp\NLP-Course\Week-02\saved_models/best_lstm_model.pt
Epoch [2/4] | Train Loss: 0.4054 | Val Loss: 0.4104
--> Saved Best Model to d:\AI\OmarNlp\NLP-Course\Week-02\saved_models/best_lstm_model.pt
Epoch [3/4] | Train Loss: 0.3199 | Val Loss: 0.3794
--> Saved Best Model to d:\AI\OmarNlp\NLP-Course\Week-02\saved_models/best_lstm_model.pt
Epoch [4/4] | Train Loss: 0.2658 | Val Loss: 0.3799
[4/4] Training Complete.

# Evaluation

evaluate.py
[1/3] Loading Artifacts...
[2/3] Loading Model...
[3/3] Generating Predictions & Computing Optimal Thresholds...

==================================================================================
Label            | Best Thresh  | Precision  | Recall     | F1-Score   | ROC-AUC   
==================================================================================
toxic            | 0.92         | 0.8288     | 0.7177     | 0.7693     | 0.9664    
severe_toxic     | 0.94         | 0.3292     | 0.7022     | 0.4482     | 0.9860    
obscene          | 0.92         | 0.7715     | 0.7920     | 0.7816     | 0.9800    
threat           | 0.94         | 0.0723     | 0.7547     | 0.1320     | 0.9825    
insult           | 0.92         | 0.6742     | 0.7500     | 0.7101     | 0.9777    
identity_hate    | 0.94         | 0.1875     | 0.4758     | 0.2690     | 0.9564    
==================================================================================
