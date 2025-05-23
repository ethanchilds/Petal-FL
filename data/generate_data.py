# FROM ChatGPT. PLACEHOLDER FOR DEVELOPMENT AND TESTING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

# Set seed for reproducibility
np.random.seed(42)

# ------------------------------
# Dataset A: Linear relationship + noise
# ------------------------------
X_a = np.random.rand(100, 2)  # 100 samples, 2 features
y_a = 3 * X_a[:, 0] + 2 * X_a[:, 1] + 1 + np.random.randn(100) * 0.1  # Linear + noise

# ------------------------------
# Dataset B: Nonlinear relationship + noise
# ------------------------------
X_b = np.random.rand(100, 2)
y_b = np.sin(5 * X_b[:, 0]) + 0.5 * X_b[:, 1]**2 + np.random.randn(100) * 0.1

# ------------------------------
# Save both to CSV
# ------------------------------
df_a = pd.DataFrame(np.hstack([X_a, y_a.reshape(-1, 1)]), columns=["feature1", "feature2", "target"])
df_b = pd.DataFrame(np.hstack([X_b, y_b.reshape(-1, 1)]), columns=["feature1", "feature2", "target"])

df_a.to_csv("fl_app/data/dataset_1.csv", index=False)
df_b.to_csv("fl_app/data/dataset_2.csv", index=False)

# ------------------------------
# Optional: Load CSVs back into PyTorch TensorDataset
# ------------------------------

'''
def load_dataset_from_csv(path):
    df = pd.read_csv(path)
    X = torch.tensor(df[["feature1", "feature2"]].values, dtype=torch.float32)
    y = torch.tensor(df["target"].values, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X, y)

# Example usage
dataset_a = load_dataset_from_csv("dataset_a.csv")
dataset_b = load_dataset_from_csv("dataset_b.csv")

print("Loaded datasets with shapes:")
print("  dataset_a:", dataset_a.tensors[0].shape)
print("  dataset_b:", dataset_b.tensors[0].shape)
'''