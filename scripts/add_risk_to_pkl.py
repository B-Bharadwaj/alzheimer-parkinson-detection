import pandas as pd

# Define the mapping
risk_map = {
    'alzheimer': 0.9,
    'parkinson': 0.8,
    'normal': 0.1
}

for split in ['train', 'val', 'test']:
    path = f"data/embedded/{split}.pkl"
    df = pd.read_pickle(path)
    df['risk'] = df['label'].map(risk_map)
    df.to_pickle(path)

print("✅ Risk scores added to all embedded datasets.")
