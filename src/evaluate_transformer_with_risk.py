import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from models.transformer_with_risk import TransformerWithRisk
from src.train_bilstm_with_risk import collate_fn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from models.cnn_bilstm_with_risk import CNNBiLSTMWithRisk
import sys, os
# add project root (one level up) to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_with_padding(path, label_encoder):
    df = pd.read_pickle(path)

    # — Add this mapping here! —
    df['risk'] = df['label'].map({
        'alzheimer': 0.9,
        'parkinson': 0.8,
        'normal': 0.1
    })

    # Extract embeddings (variable-length sequences)
    X = df['embedding'].tolist()
    labels = label_encoder.transform(df['label'])
    risks = df['risk'].values

    # Compute lengths before padding
    lengths = torch.tensor([len(x) for x in X], dtype=torch.long)

    # Determine the maximum sequence length
    max_len = max(lengths)

    # Pad each sequence to the maximum length
    padded_X = [np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant') for seq in X]
    X_tensor = torch.tensor(np.array(padded_X), dtype=torch.float32)

    # Convert labels and risks to tensors
    y_tensor = torch.tensor(labels, dtype=torch.long)
    r_tensor = torch.tensor(risks, dtype=torch.float32)

    # Return TensorDataset and original string labels
    dataset = TensorDataset(X_tensor, y_tensor, r_tensor, lengths)
    return dataset, df['label'].tolist()

    # # Pad sequences manually
    # padded_X = [np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant') for seq in X]
    # X = torch.tensor(np.array(padded_X), dtype=torch.float32)
    # y = torch.tensor(labels, dtype=torch.long)
    # r = torch.tensor(risks, dtype=torch.float32)
    # lengths = torch.tensor(lengths, dtype=torch.long)

    # return TensorDataset(X, y, r, lengths), df['label'].tolist()

if __name__ == "__main__":
    df_train = pd.read_pickle("data/embedded/train.pkl")
    le = LabelEncoder()
    le.fit(df_train['label'])

    dataset, true_labels_str = load_data_with_padding("data/embedded/test.pkl", le)
    dataloader = DataLoader(dataset, batch_size=8)

    model = TransformerWithRisk()
    model.load_state_dict(torch.load("saved_model/transformer_with_risk.pt", map_location=device))
    model.to(device)
    model.eval()

    model = CNNBiLSTMWithRisk().to(device)
    model.load_state_dict(torch.load("saved_model/cnn_bilstm_with_risk.pt", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    all_risk_preds = []
    all_risk_true = []

    with torch.no_grad():
        for xb, yb, rb, lengths in dataloader:
            xb, yb, rb, lengths = xb.to(device), yb.to(device), rb.to(device), lengths.to(device)
            logits, risk_pred = model(xb, lengths)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_risk_preds.extend(risk_pred.cpu().numpy())
            all_risk_true.extend(rb.cpu().numpy())

    # Evaluation
    print("✅ Evaluation Complete")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))
    print(confusion_matrix(all_labels, all_preds))

    mse = mean_squared_error(all_risk_true, all_risk_preds)
    print(f"\n🧠 Misfolding Risk MSE: {mse:.4f}")
