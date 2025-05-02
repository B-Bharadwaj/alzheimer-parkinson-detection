import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from train_bilstm_with_risk import BiLSTMWithRisk, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProteinDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.embeddings = [torch.tensor(e, dtype=torch.float32) for e in df['embedding']]
        self.labels = torch.tensor(label_encoder.transform(df['label']), dtype=torch.long)
        self.risks = torch.tensor(df['label'].map({
            'alzheimer': 0.9,
            'parkinson': 0.8,
            'normal': 0.1
        }).values, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.risks[idx]

def evaluate_model(model, dataloader, label_encoder):
    model.eval()
    all_preds, all_labels, all_risk_preds, all_risk_true = [], [], [], []

    with torch.no_grad():
        for xb, yb, risk_true, lengths in dataloader:
            xb, yb, risk_true, lengths = xb.to(device), yb.to(device), risk_true.to(device), lengths.to(device)
            class_logits, risk_preds = model(xb, lengths)

            all_preds.extend(class_logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(yb.cpu().tolist())
            all_risk_preds.extend(risk_preds.cpu().tolist())
            all_risk_true.extend(risk_true.cpu().tolist())

    print("✅ Evaluation Complete")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    print(confusion_matrix(all_labels, all_preds))

    mse = mean_squared_error(all_risk_true, all_risk_preds)
    print(f"\n🧠 Misfolding Risk MSE: {mse:.4f}")

    # Optional: Plot predicted vs true risk
    plt.scatter(all_risk_true, all_risk_preds, c='blue', alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("True Risk")
    plt.ylabel("Predicted Risk")
    plt.title("Predicted vs True Misfolding Risk")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_test = pd.read_pickle("data/embedded/test.pkl")
    df_train = pd.read_pickle("data/embedded/train.pkl")
    le = LabelEncoder()
    le.fit(df_train['label'])

    test_dataset = ProteinDataset(df_test, le)
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

    model = BiLSTMWithRisk().to(device)
    model.load_state_dict(torch.load("model/bilstm_with_risk.pt", map_location=device))

    evaluate_model(model, test_loader, le)
