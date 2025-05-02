import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from train_bilstm_classifier import BiLSTMClassifier, collate_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProteinDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.embeddings = [torch.tensor(e, dtype=torch.float32) for e in df['embedding']]
        self.labels = torch.tensor(label_encoder.transform(df['label']), dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def evaluate_model(model, dataloader, label_encoder):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb, lengths in dataloader:
            xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
            outputs = model(xb, lengths)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(yb.cpu().tolist())

    print("✅ Evaluation Complete")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    df_test = pd.read_pickle("data/embedded/test.pkl")

    # Encode labels
    df_train = pd.read_pickle("data/embedded/train.pkl")
    label_encoder = LabelEncoder()
    label_encoder.fit(df_train['label'])

    test_dataset = ProteinDataset(df_test, label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

    # Load trained BiLSTM model
    model = BiLSTMClassifier().to(device)
    model.load_state_dict(torch.load("model/bilstm_classifier.pt", map_location=device))

    evaluate_model(model, test_loader, label_encoder)
