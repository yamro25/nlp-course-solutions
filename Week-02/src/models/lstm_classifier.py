import torch
import torch.nn as nn

class ToxicLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int = 6):
        super(ToxicLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=False,  # أسرع بكثير على الـ CPU
            num_layers=1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),  # بدون ضرب في 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        # Global Max Pooling over sequence dimension
        pooled = torch.max(out, dim=1)[0]
        logits = self.fc(pooled)
        return logits