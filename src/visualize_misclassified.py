# src/visualize_misclassified.py

import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from train_bilstm_with_risk import BiLSTMWithRisk, collate_fn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_pickle("data/embedded/test.pkl")
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])

X = torch.tensor(df['embedding'].tolist()).float()
y = torch.tensor(label_encoder.transform(df['label']))
risk_true = torch.tensor(df['risk'].tolist()).float()
dataset = TensorDataset(X, y, risk_true)
loader = DataLoader(dataset, batch_size=1)

# Load model
model = BiLSTMWithRisk().to(device)
model.load_state_dict(torch.load("model/bilstm_with_risk.pt"))
model.eval()

# Predict and collect
misclassified = []
with torch.no_grad():
    for i, (xb, yb, riskb) in enumerate(loader):
        xb, yb, riskb = xb.to(device), yb.to(device), riskb.to(device)
        logits, risk_pred = model(xb, torch.tensor([xb.size(1)]).to(device))
        pred_class = logits.argmax(dim=1).item()
        true_class = yb.item()

        if pred_class != true_class:
            misclassified.append({
                "True": label_encoder.inverse_transform([true_class])[0],
                "Predicted": label_encoder.inverse_transform([pred_class])[0],
                "True Risk": float(riskb.item()),
                "Predicted Risk": float(risk_pred.item()),
                "Sequence (start)": df.iloc[i]['sequence'][:30]
            })

pd.DataFrame(misclassified).to_csv("misclassified.csv", index=False)
print("✅ Misclassified sequences saved to misclassified.csv")
