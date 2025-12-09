# 🧬 Alzheimer & Parkinson’s Detection Using Protein Sequences  
A deep learning project for classifying **Alzheimer’s**, **Parkinson’s**, and **Normal** protein sequences using **TAPE embeddings** and modern neural architectures.

---

## 🌟 Overview  
This project focuses on detecting neurodegenerative diseases from **protein primary sequences**.  
Instead of using clinical or imaging data, the model learns **sequence-level patterns** associated with disease-linked misfolding or dysfunction.

The system currently performs **3-class classification**:

- **Alzheimer’s disease**  
- **Parkinson’s disease**  
- **Normal proteins**

---

## 🧠 Key Features

### **✔ TAPE Protein Embeddings**  
- Pretrained transformer-based protein embeddings  
- Captures biochemical, structural, and evolutionary features of sequences  
- Provides contextual representation for each amino acid token  

### **✔ Multiple Deep Learning Architectures**  
Implemented architectures include:

- **BiLSTM Model**  
- **BiLSTM + Attention** (Luong-style attention for interpretability)  
- **Transformer-based Classifier** (custom encoder with positional encoding)

### **✔ Misfolding Risk Regression (Auxiliary Head)**  
- Optional regression output predicting **misfolding risk score**  
- Helps interpret protein stability and folding behavior  
- Provides richer biological insights alongside classification

### **✔ Visualization & Evaluation Tools**  
- **Attention heatmaps** for misclassified sequences  
- **Embedding projections** (t-SNE, PCA)  
- **Training curves** (loss & accuracy)  
- Detailed evaluation metrics (Accuracy, Precision, Recall, F1)

---

## 📊 Model Comparison

| Model                | Accuracy    | Misfolding Risk MSE | Notes                          |
|---------------------|-------------|----------------------|-------------------------------|
| MLP (baseline)       | ~0.50       | 0.20+                | Shallow, not position-aware   |
| BiLSTM + Risk        | 0.65–0.83   | 0.09–0.12            | Sequence-aware                |
| BiLSTM + Attention   | 0.70+       | 0.08–0.11            | Better focus on key tokens    |
| Transformer + Risk   | 0.67–0.82   | 0.10–0.14            | Stronger on longer sequences  |
| CNN-BiLSTM + Risk    | ~0.65       | ~0.11                | Feature extraction + memory   |


---

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
```

## 📊 Results Summary

This section showcases evaluation metrics for different models used to classify protein sequences into **Alzheimer**, **Parkinson**, and **Normal** classes. Additionally, each model predicts a **misfolding risk score** (regression).

---

### ✅ Model Comparison Table

| Model            | Alzheimer F1 | Parkinson F1 | Normal F1 | Risk MSE |
|------------------|---------------|--------------|-----------|----------|
| Transformer      | 0.75          | 0.50         | 0.67      | 0.1099   |
| CNN-BiLSTM       | 0.67          | 0.53         | 0.60      | 0.118x   |
| BiLSTM + Risk    | 0.65          | 0.47         | 0.60      | 0.131x   |
| CNN-BiLSTM + Aug | 0.71          | 0.50         | 0.63      | 0.1099   |


