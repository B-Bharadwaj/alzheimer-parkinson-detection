import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNBiLSTMWithRisk(nn.Module):
    def __init__(self, embed_dim=768, num_classes=3, cnn_filters=128, lstm_hidden=128, dropout=0.3):
        super().__init__()
        # 1D CNN motif extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden*2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        # Risk regression head
        self.risk_head = nn.Sequential(
            nn.Linear(lstm_hidden*2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths):
        # x: (batch, seq_len, embed_dim)
        x = x.transpose(1,2)  # → (batch, embed_dim, seq_len)
        x = self.cnn(x)      # → (batch, cnn_filters, seq_len')
        x = x.transpose(1,2)  # → (batch, seq_len', cnn_filters)

        # Pack & LSTM
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Compute new lengths after two MaxPool1d(2) layers
        # Each pooling halves seq_len (floor division)
        down1 = torch.floor_divide(lengths, 2)
        down2 = torch.floor_divide(down1,   2)
        new_lengths = down2.clamp(min=1)     # ensure at least 1

        # Pack padded sequence using downsampled lengths
        packed = pack_padded_sequence(
           x, new_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)  # → (batch, seq_len', 2*lstm_hidden)

        # Mask padding & mean pool
        mask = (torch.arange(out.size(1), device=lengths.device)
                .unsqueeze(0) < lengths.unsqueeze(1))
        mask = torch.arange(out.size(1), device=lengths.device).unsqueeze(0) < new_lengths.unsqueeze(1)
        masked = out * mask.unsqueeze(-1)
        pooled = masked.sum(1) / lengths.unsqueeze(1)
        pooled = masked.sum(1) / new_lengths.unsqueeze(1)
 
        logits    = self.classifier(pooled)
        risk_score = self.risk_head(pooled).squeeze(1)
        return logits, risk_score
