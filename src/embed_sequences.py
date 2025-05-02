import pandas as pd
import torch
import os
from tape import ProteinBertModel, TAPETokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load TAPE BERT model and tokenizer
tokenizer = TAPETokenizer(vocab='iupac')
model = ProteinBertModel.from_pretrained('bert-base').to(device)
model.eval()

# Extract full token-level embeddings for BiLSTM (shape: [seq_len, 768])
def get_token_embeddings(seq):
    token_ids = torch.tensor([tokenizer.encode(seq)]).to(device)
    with torch.no_grad():
        output = model(token_ids)  # output[0] = [1, seq_len, 768]
        return output[0].squeeze(0).cpu().numpy()  # shape: [seq_len, 768]

# Process and embed dataset
def embed_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    df = df[df['sequence'].apply(lambda x: isinstance(x, str) and len(x) >= 10)]

    if len(df) == 0:
        print(f"⚠️  No valid sequences found in {file_path}")
        return

    print(f"Embedding {len(df)} sequences from {file_path}")
    try:
        df['embedding'] = df['sequence'].apply(get_token_embeddings)
        df.to_pickle(output_path)
        print(f"✅ Saved {len(df)} sequences to {output_path}")
    except Exception as e:
        print("❌ Error while embedding:", e)

if __name__ == "__main__":
    os.makedirs("data/embedded", exist_ok=True)
    embed_dataset("data/train.csv", "data/embedded/train.pkl")
    embed_dataset("data/val.csv", "data/embedded/val.pkl")
    embed_dataset("data/test.csv", "data/embedded/test.pkl")
