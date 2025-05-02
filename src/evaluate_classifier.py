import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from train_classifier import ProteinClassifier, load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_pickle("data/embedded/train.pkl")
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])

test_data = load_dataset("data/embedded/test.pkl", label_encoder)
test_loader = DataLoader(test_data, batch_size=8)

model = ProteinClassifier().to(device)
model.load_state_dict(torch.load("model/classifier.pt"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1).cpu()
        all_preds.extend(preds)
        all_labels.extend(yb)

print("✅ Evaluation Complete")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
print(confusion_matrix(all_labels, all_preds))
