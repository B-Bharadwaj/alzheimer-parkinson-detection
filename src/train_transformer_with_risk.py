import sys, os
# make sure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from models.cnn_bilstm_with_risk import CNNBiLSTMWithRisk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

# === Data Loader with Padding & Risk Injection ===
def load_data_with_padding(path, label_encoder):
    df = pd.read_pickle(path)
    # add risk column
    df['risk'] = df['label'].map({'alzheimer': 0.9, 'parkinson': 0.8, 'normal': 0.1})

    X = df['embedding'].tolist()
    y = label_encoder.transform(df['label'])
    r = df['risk'].values

    lengths = torch.tensor([len(seq) for seq in X], dtype=torch.long)
    max_len = int(lengths.max())

    # pad to max_len
    padded = [np.pad(seq, ((0, max_len - len(seq)), (0,0)), mode='constant') for seq in X]
    X_tensor = torch.tensor(np.stack(padded), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    r_tensor = torch.tensor(r, dtype=torch.float32)

    return TensorDataset(X_tensor, y_tensor, r_tensor, lengths)

if __name__ == "__main__":
    # --- Prepare label encoder on train split ---
    df_train = pd.read_pickle("data/embedded/train.pkl")
    le = LabelEncoder().fit(df_train['label'])

    # --- Load datasets ---
    train_ds = load_data_with_padding("data/embedded/train.pkl", le)
    train_ds = load_data_with_padding("data/embedded/train_augmented.pkl", le)
    val_ds   = load_data_with_padding("data/embedded/val.pkl",   le)
    val_ds   = load_data_with_padding("data/embedded/val_augmented.pkl", le)

    # --- Oversample minority classes ---
    labels = train_ds.tensors[1].cpu().numpy()
    uniq, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(uniq, counts))
    weights = np.array([1.0 / class_counts[int(l)] for l in labels])
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)

    # --- Model, losses, optimizer, scheduler ---
    model = CNNBiLSTMWithRisk().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    class_w = compute_class_weight("balanced",
                                   classes=le.classes_,
                                   y=df_train['label'])
    class_w = torch.tensor(class_w, dtype=torch.float32).to(device)

    clf_loss_fn  = FocalLoss(alpha=class_w)
    risk_loss_fn = nn.MSELoss()

    # --- Training loop with early stopping ---
    train_losses, val_losses = [], []
    best_val, patience, counter = float('inf'), 5, 0

    for epoch in range(1, 51):
        model.train()
        t_clf, t_risk, correct, total = 0, 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}/50")

        for xb, yb, rb, lengths in loop:
            xb, yb, rb, lengths = xb.to(device), yb.to(device), rb.to(device), lengths.to(device)
            optimizer.zero_grad()
            logits, risk = model(xb, lengths)
            loss = clf_loss_fn(logits, yb) + risk_loss_fn(risk, rb)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            t_clf  += clf_loss_fn(logits, yb).item()
            t_risk += risk_loss_fn(risk, rb).item()

            loop.set_postfix(acc=f"{correct/total:.2%}",
                             clf=f"{t_clf:.4f}",
                             risk=f"{t_risk:.4f}")

        scheduler.step()

        # --- Validation ---
        model.eval()
        v_clf, v_risk = 0, 0
        with torch.no_grad():
            for xb, yb, rb, lengths in val_loader:
                xb, yb, rb, lengths = xb.to(device), yb.to(device), rb.to(device), lengths.to(device)
                logits, risk = model(xb, lengths)
                v_clf  += clf_loss_fn(logits, yb).item()
                v_risk += risk_loss_fn(risk, rb).item()

        val_loss = v_clf + v_risk
        train_losses.append(t_clf + t_risk)
        val_losses.append(val_loss)

        train_acc = correct / total
        print(f"Epoch {epoch:02d} | TrainLoss: {t_clf+t_risk:.4f} | ValLoss: {val_loss:.4f} | Acc: {train_acc:.2%}")

        # early stopping & save
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs("saved_model", exist_ok=True)
            torch.save(model.state_dict(), "saved_model/cnn_bilstm_with_risk.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch:02d}")
                break

    # --- Plot losses ---
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN+BiLSTM with Risk Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("transformer_training_loss.png")
    print("📉 Saved loss plot")
