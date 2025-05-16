# models/transformer_with_risk.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2500):  # 🔼 was 512
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Input sequence length {seq_len} exceeds max_len {self.pe.size(1)} in PositionalEncoding.")
        x = x + self.pe[:, :seq_len].to(x.device)
        return self.dropout(x)


class TransformerWithRisk(nn.Module):
    def __init__(self, input_dim=768, num_classes=3, risk_out=1, nhead=8, dim_feedforward=2048, nlayers=2):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.risk_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, risk_out)
        )

    def forward(self, x, lenghts =None):
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling

        class_logits = self.classifier(x)
        risk_score = self.risk_head(x)
        return class_logits, risk_score.squeeze(1)
