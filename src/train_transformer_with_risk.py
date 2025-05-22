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


    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []


    # --- Training loop with early stopping ---
    train_losses, val_losses = [], []
    best_val, patience, counter = float('inf'), 5, 0

    for epoch in range(1, 51):
        model.train()
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}/50")

        for xb, yb, rb, lengths in loop:
            xb, yb, rb, lengths = xb.to(device), yb.to(device), rb.to(device), lengths.to(device)
            optimizer.zero_grad()
            logits, risk = model(xb, lengths)
            loss = clf_loss_fn(logits, yb) + risk_loss_fn(risk, rb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            loop.set_postfix(acc=f"{correct/total:.2%}", loss=loss.item())

        scheduler.step()

        train_acc = correct / total
        train_losses.append(total_loss)
        train_accuracies.append(train_acc)

    # === Validation ===
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb, rb, lengths in val_loader:
                xb, yb, rb, lengths = xb.to(device), yb.to(device), rb.to(device), lengths.to(device)
                logits, risk = model(xb, lengths)
                loss = clf_loss_fn(logits, yb) + risk_loss_fn(risk, rb)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_losses.append(val_loss)
        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch:02d} | TrainLoss: {total_loss:.4f} | ValLoss: {val_loss:.4f} | TrainAcc: {train_acc:.2%} | ValAcc: {val_acc:.2%}")

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
    import matplotlib.pyplot as plt

    os.makedirs("images", exist_ok=True)

    # Plot: Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer with Risk - Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/transformer_loss_curve.png")
    plt.close()

    # Plot: Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Transformer with Risk - Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/transformer_accuracy_curve.png")
    plt.close()

    print("✅ Plots saved to images/transformer_loss_curve.png and transformer_accuracy_curve.png")
