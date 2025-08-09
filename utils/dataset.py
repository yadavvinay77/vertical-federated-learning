# utils/dataset.py
import numpy as np
import pandas as pd

class ClientDataset:
    """
    Loads client data from CSV, normalizes numeric features,
    and exposes features and labels as numpy arrays.
    """
    def __init__(self, path):
        df = pd.read_csv(path)
        
        # Find label columns (columns containing 'income')
        label_cols = [col for col in df.columns if "income" in col]
        
        # Feature columns = all except label columns
        feature_cols = [col for col in df.columns if col not in label_cols]
        
        # Select numeric feature columns only for normalization
        numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Compute mean and std deviation for normalization
        self.means = df[numeric_cols].mean()
        self.stds = df[numeric_cols].std().replace(0, 1)  # avoid division by zero
        
        # Normalize numeric features
        df[numeric_cols] = (df[numeric_cols] - self.means) / self.stds
        
        # Store features as numpy array
        self.features = df[feature_cols].values
        
        # Store labels as numpy array (assumes single income label column)
        self.labels = df[label_cols[0]].values if label_cols else None
        
        # Number of samples
        self.n_samples = len(df)
