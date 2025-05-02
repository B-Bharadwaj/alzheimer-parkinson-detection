# 🧠 Protein Misfolding Classifier (Alzheimer's, Parkinson's, Normal)

This project uses protein structure sequences to classify whether a protein is associated with Alzheimer's, Parkinson's, or is Normal.  
It also predicts a **misfolding risk score** for each sequence using **token-level embeddings** from TAPE and a **BiLSTM-based neural network**.

---

## 📁 Project Structure

A&P_Detection/
├── data/ # Contains train/val/test datasets and raw .pdb files
├── model/ # Saved PyTorch models
├── src/ # All training, evaluation, embedding scripts
├── requirements.txt # Python dependencies
└── README.md # This file



---

## 🧪 Model Comparison

| Model                | Accuracy | Notes                                |
|----------------------|----------|--------------------------------------|
| `MLP + CLS`          | 66–70%   | Uses [CLS] token from TAPE embeddings only |
| `BiLSTM`             | 83%      | Uses full token-level embeddings     |
| `BiLSTM + Risk`      | 83% + regression | Predicts both class and misfolding risk |

---

## 🔧 Setup Instructions

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## 🧬 How to Use

### 1. Convert `.pdb` files to sequence
```bash
python src/extract_sequences.py

2. Generate TAPE embeddings
python src/embed_sequences.py

3.Train a model
python src/train_classifier.py                # MLP + [CLS]
python src/train_bilstm_classifier.py         # BiLSTM with token-level embeddings
python src/train_bilstm_with_risk.py          # BiLSTM + Risk Score (multi-task)

4. Evaluate models
python src/evaluate_classifier.py             # Evaluate MLP
python src/evaluate_bilstm_classifier.py      # Evaluate BiLSTM
python src/evaluate_bilstm_with_risk.py       # Evaluate BiLSTM + Risk + plot

5. (Optional) Visualize Embeddings
python src/visualize_tsne.py


