import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, encoder_outputs, lengths):
        # encoder_outputs: [batch, seq_len, hidden*2]
        energy = torch.tanh(self.attn(encoder_outputs))  # [batch, seq_len, hidden]
        energy = torch.matmul(energy, self.v)  # [batch, seq_len]
        
        # Mask out padded tokens
        mask = torch.arange(encoder_outputs.size(1), device=lengths.device)[None, :] >= lengths[:, None]
        energy.masked_fill_(mask, -1e9)

        attn_weights = torch.softmax(energy, dim=1)  # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch, hidden*2]
        return context, attn_weights

class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        context, attn_weights = self.attention(outputs, lengths)
        logits = self.classifier(context)
        risk = self.risk_head(context).squeeze(1)
        return logits, risk, attn_weights
