# 🧬 Alzheimer's & Parkinson's Protein Misfolding Detection

This project detects Alzheimer's, Parkinson's, and normal protein sequences based on misfolding risk using deep learning. It uses TAPE embeddings, BiLSTM + attention, and Transformer architectures, trained on `.pdb` sequences derived from Homo sapiens proteins.

---

## 📊 Model Comparison

| Model                | Accuracy    | Misfolding Risk MSE | Notes                          |
|---------------------|-------------|----------------------|-------------------------------|
| MLP (baseline)       | ~0.50       | 0.20+                | Shallow, not position-aware   |
| BiLSTM + Risk        | 0.65–0.83   | 0.09–0.12            | Sequence-aware                |
| BiLSTM + Attention   | 0.70+       | 0.08–0.11            | Better focus on key tokens    |
| Transformer + Risk   | 0.67–0.82   | 0.10–0.14            | Stronger on longer sequences  |
| CNN-BiLSTM + Risk    | ~0.65       | ~0.11                | Feature extraction + memory   |




## 📂 Project Structure

```
A&P_Detection/
├── data/                     # CSV and embedded .pkl files
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── embedded/
│       ├── train.pkl
│       ├── val.pkl
│       └── test.pkl
│
├── models/                  # Model architecture definitions
│   ├── cnn_bilstm_with_risk.py
│   ├── transformer_with_risk.py
│   └── bilstm_with_attention.py
│
├── saved_model/             # Trained model weights
│   ├── cnn_bilstm_with_risk.pt
│   ├── transformer_with_risk.pt
│   └── bilstm_with_risk.pt
│
├── scripts/                 # Preprocessing utilities
│   ├── add_risk_to_pkl.py
│   └── augment_with_windows.py
│
├── src/                     # Main training, evaluation, visualization
│   ├── train_bilstm_with_risk.py
│   ├── train_transformer_with_risk.py
│   ├── evaluate_bilstm_with_risk.py
│   ├── evaluate_transformer_with_risk.py
│   └── visualize_misclassified.py
│
├── requirements.txt
└── README.md
```


---

## 🧠 How to Use 
```bash
1.Extract Sequences from `.pdb` files
python src/extract_sequences.py

2. Generate TAPE Embeddings
python src/embed_sequences.py

3. Add Risk Scores (if missing)
python scripts/add_risk_to_pkl.py

4. Train Models
🔁 Train BiLSTM
python src/train_bilstm_with_risk.py

🧠 Train Transformer
python src/train_transformer_with_risk.py

🧬 CNN + BiLSTM Hybrid
python src/train_cnn_bilstm_with_risk.py

5. Evaluate Models
python src/evaluate_bilstm_with_risk.py
python src/evaluate_transformer_with_risk.py

---


## 📊 Results Summary
```
This section showcases evaluation metrics for different models used to classify protein sequences into **Alzheimer**, **Parkinson**, and **Normal** classes. Additionally, each model predicts a **misfolding risk score** (regression).

---

### ✅ Model Comparison Table

| Model            | Accuracy | Alzheimer F1 | Parkinson F1 | Normal F1 | Risk MSE |
|------------------|----------|---------------|--------------|-----------|----------|
| Transformer      | 65%      | 0.75          | 0.50         | 0.67      | 0.1099   |
| CNN-BiLSTM       | 63%      | 0.67          | 0.53         | 0.60      | 0.118x   |
| BiLSTM + Risk    | 60%      | 0.65          | 0.47         | 0.60      | 0.131x   |
| CNN-BiLSTM + Aug | 65%      | 0.71          | 0.50         | 0.63      | 0.1099   |
```

### 📉 Training vs Validation Loss (Transformer)

![Loss Curve](https://raw.githubusercontent.com/B-Bharadwaj/alzheimer-parkinson-detection/main/images/transformer_training_loss_v2.png)

---

### 🔄 Confusion Matrices (Visual)

#### ✅ Transformer

![Transformer Confusion Matrix](https://raw.githubusercontent.com/B-Bharadwaj/alzheimer-parkinson-detection/main/images/confusion_matrix_transformer_v2.png)

---

#### ✅ CNN-BiLSTM + Augmentation

![CNN-BiLSTM Confusion Matrix](https://raw.githubusercontent.com/B-Bharadwaj/alzheimer-parkinson-detection/main/images/confusion_matrix_cnn_bilstm_v2.png)

---

#### ✅ BiLSTM + Risk

![BiLSTM + Risk Confusion Matrix](https://raw.githubusercontent.com/B-Bharadwaj/alzheimer-parkinson-detection/main/images/confusion_matrix_bilstm_risk_v2.png)
