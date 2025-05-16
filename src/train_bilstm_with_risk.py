# train_bilstm_with_risk.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bilstm_with_attention import BiLSTMWithAttention
#from models.transformer_with_risk import TransformerWithRisk
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from models.bilstm_with_attention import BiLSTMWithAttention
from tqdm import tqdm
import matplotlib.pyplot as plt
#model = TransformerWithRisk().to(device)
class BiLSTMWithRisk(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        self.risk_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, x, lengths):
        lstm_out, _ = self.lstm(x)
        out = lstm_out.mean(dim=1)
        return self.classifier(out), self.risk_head(out).squeeze(1)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels, risks = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(risks, dtype=torch.float32), lengths


def load_dataset(path, label_encoder):
    df = pd.read_pickle(path)
    df['risk'] = df['label'].map({'alzheimer': 0.9, 'parkinson': 0.8, 'normal': 0.1})
    X = [torch.tensor(x, dtype=torch.float32) for x in df['embedding']]
    y = label_encoder.transform(df['label'])
    risks = df['risk'].tolist()
    return list(zip(X, y, risks))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train = pd.read_pickle("data/embedded/train.pkl")
    df_val = pd.read_pickle("data/embedded/val.pkl")

    label_encoder = LabelEncoder()
    label_encoder.fit(df_train['label'])

    train_data = load_dataset("data/embedded/train.pkl", label_encoder)
    val_data = load_dataset("data/embedded/val.pkl", label_encoder)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=8, collate_fn=collate_fn)

    model = BiLSTMWithRisk().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=label_encoder.classes_,
                                         y=df_train['label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    classification_loss = nn.CrossEntropyLoss(weight=class_weights)
    risk_loss = nn.MSELoss()

    best_val_loss = float("inf")
    patience = 7
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(100):
        model.train()
        total_clf_loss, total_risk_loss, correct, total = 0, 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/50", leave=False)

        for xb, yb, risk_target, lengths in loop:
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

            loop.set_postfix({
                "acc": f"{(correct / total):.2%}",
                "clf_loss": f"{loss1.item():.4f}",
                "risk_loss": f"{loss2.item():.4f}"
            })

        scheduler.step()
        train_losses.append(total_clf_loss + total_risk_loss)

        # Validation loop
        model.eval()
        val_clf_loss, val_risk_loss = 0, 0
        with torch.no_grad():
            for xb, yb, risk_target, lengths in val_loader:
                xb, yb, risk_target, lengths = xb.to(device), yb.to(device), risk_target.to(device), lengths.to(device)
                class_logits, risk_pred = model(xb, lengths)
                val_clf_loss += classification_loss(class_logits, yb).item()
                val_risk_loss += risk_loss(risk_pred, risk_target).item()

        val_total_loss = val_clf_loss + val_risk_loss
        val_losses.append(val_total_loss)

        acc = correct / total
        print(f"Epoch {epoch+1:02d} | Clf Loss: {total_clf_loss:.4f} | Risk Loss: {total_risk_loss:.4f} | Val Loss: {val_total_loss:.4f} | Acc: {acc:.2%}")

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            os.makedirs("model", exist_ok=True)  # ← Add this line
            torch.save(model.state_dict(), "model/bilstm_with_risk.pt")
            counter = 0

        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"⏹️ Early stopping at epoch {epoch+1}")
        #         break

    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_vs_validation_loss.png")
    print("📉 Saved loss plot to training_vs_validation_loss.png")
