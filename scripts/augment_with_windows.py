import pandas as pd
import numpy as np
import os

# Map labels to risk scores
RISK_MAP = {
    'alzheimer': 0.9,
    'parkinson': 0.8,
    'normal': 0.1
}

def sliding_windows(arr, window_size=100, stride=50):
    """Yield sub-arrays of shape (window_size, embed_dim)."""
    L, D = arr.shape
    for start in range(0, L - window_size + 1, stride):
        yield arr[start:start + window_size]

def augment_split(split):
    in_path  = f"data/embedded/{split}.pkl"
    out_path = f"data/embedded/{split}_augmented.pkl"
    df = pd.read_pickle(in_path)
    aug_records = []

    for _, row in df.iterrows():
        emb   = row['embedding']   # numpy array (L, 768)
        label = row['label']
        risk  = RISK_MAP[label]    # compute risk here

        if emb.shape[0] < 100:
            # too short → keep whole sequence
            aug_records.append({'embedding': emb, 'label': label, 'risk': risk})
        else:
            # slide windows
            for win in sliding_windows(emb, window_size=100, stride=50):
                aug_records.append({'embedding': win, 'label': label, 'risk': risk})

    out_df = pd.DataFrame(aug_records)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_pickle(out_path)
    print(f"✅ {split}: {len(out_df)} windows saved to {out_path}")

if __name__ == "__main__":
    for split in ['train', 'val']:
        augment_split(split)
