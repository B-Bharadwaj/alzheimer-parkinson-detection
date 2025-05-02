import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProteinDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.embeddings = [torch.tensor(e, dtype=torch.float32) for e in df['embedding']]
        self.labels = torch.tensor(label_encoder.transform(df['label']), dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.stack(labels), lengths

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (h_n, _) = self.lstm(packed)
        hidden = torch.cat((h_n[0], h_n[1]), dim=1)  # shape: [batch, hidden*2]
        return self.fc(hidden)

class BiLSTMWithRisk(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # outputs between 0 and 1
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (h_n, _) = self.lstm(packed)
        hidden = torch.cat((h_n[0], h_n[1]), dim=1)  # shape: [batch, hidden*2]

        class_logits = self.classifier(hidden)
        risk_score = self.risk_head(hidden)

        return class_logits, risk_score


if __name__ == "__main__":
    # Load and encode
    df = pd.read_pickle("data/embedded/train.pkl")
    df['risk'] = df['label'].map({
    'alzheimer': 0.9,
    'parkinson': 0.8,
    'normal': 0.1
})
    le = LabelEncoder()
    le.fit(df['label'])

    # Dataset and DataLoader
    train_dataset = ProteinDataset(df, le)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=le.classes_, y=df['label'])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    model = BiLSTMClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train loop
    for epoch in range(30):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb, lengths in train_loader:
            xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
            optimizer.zero_grad()
            outputs = model(xb, lengths)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}, Accuracy: {acc:.2%}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/bilstm_classifier.pt")
    print("✅ BiLSTM model saved to model/bilstm_classifier.pt")
