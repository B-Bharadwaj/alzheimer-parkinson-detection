import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# Load embedded dataset
df = pd.read_pickle("data/embedded/train.pkl")  # You can also try val/test

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(df['label'])  # 0, 1, 2

# Extract 768D embeddings
import numpy as np
X = np.stack(df['embedding'].values)

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_2d = tsne.fit_transform(X)


# Plot
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
for class_index, class_label in enumerate(le.classes_):
    indices = labels == class_index
    plt.scatter([X_2d[i][0] for i in range(len(X_2d)) if indices[i]],
                [X_2d[i][1] for i in range(len(X_2d)) if indices[i]],
                label=class_label,
                alpha=0.7,
                c=colors[class_index])

plt.legend()
plt.title("t-SNE of TAPE Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.grid(True)
plt.show()
