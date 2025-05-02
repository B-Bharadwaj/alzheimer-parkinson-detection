import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProteinClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def load_dataset(path, label_encoder):
    df = pd.read_pickle(path)
    X = torch.tensor(np.array(df['embedding'].tolist())).float()
    y = torch.tensor(label_encoder.transform(df['label'])).long()
    return TensorDataset(X, y)

if __name__ == "__main__":
    # Load training data and fit label encoder
    df = pd.read_pickle("data/embedded/train.pkl")
    label_encoder = LabelEncoder()
    label_encoder.fit(df['label'])

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=label_encoder.classes_,
        y=df['label']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Load datasets
    train_data = load_dataset("data/embedded/train.pkl", label_encoder)
    val_data   = load_dataset("data/embedded/val.pkl", label_encoder)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=8)

    # Initialize model
    model = ProteinClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(100):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Accuracy: {accuracy:.2%}")

    # Save model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/classifier.pt")
    print("✅ Model saved to model/classifier.pt")
