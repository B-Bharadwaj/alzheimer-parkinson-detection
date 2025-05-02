import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class ProteinDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.embeddings = [torch.tensor(e, dtype=torch.float32) for e in df['embedding']]
        self.labels = torch.tensor(label_encoder.transform(df['label']), dtype=torch.long)
        self.risks = torch.tensor(df['risk'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.risks[idx]

def collate_fn(batch):
    sequences, labels, risks = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.stack(labels), torch.stack(risks), lengths

# BiLSTM model with classification + risk output
class BiLSTMWithRisk(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        hidden = torch.cat((h_n[0], h_n[1]), dim=1)
        class_logits = self.classifier(hidden)
        risk_score = self.risk_head(hidden).squeeze(1)
        return class_logits, risk_score

if __name__ == "__main__":
    # Load training data and simulate risk scores
    df = pd.read_pickle("data/embedded/train.pkl")
    df['risk'] = df['label'].map({
        'alzheimer': 0.9,
        'parkinson': 0.8,
        'normal': 0.1
    })

    label_encoder = LabelEncoder()
    label_encoder.fit(df['label'])

    # Dataset
    dataset = ProteinDataset(df, label_encoder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=label_encoder.classes_, y=df['label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Model
    model = BiLSTMWithRisk().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    classification_loss = nn.CrossEntropyLoss(weight=class_weights)
    risk_loss = nn.MSELoss()

    # Train
    for epoch in range(50):
        model.train()
        total_clf_loss, total_risk_loss, correct, total = 0, 0, 0, 0

        for xb, yb, risk_target, lengths in dataloader:
            xb, yb, risk_target, lengths = xb.to(device), yb.to(device), risk_target.to(device), lengths.to(device)

            optimizer.zero_grad()
            class_logits, risk_pred = model(xb, lengths)

            loss1 = classification_loss(class_logits, yb)
            loss2 = risk_loss(risk_pred, risk_target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            total_clf_loss += loss1.item()
            total_risk_loss += loss2.item()
            preds = class_logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        print(f"Epoch {epoch+1} - Clf Loss: {total_clf_loss:.4f}, Risk Loss: {total_risk_loss:.4f}, Accuracy: {correct/total:.2%}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/bilstm_with_risk.pt")
    print("✅ BiLSTM+Risk model saved to model/bilstm_with_risk.pt")
