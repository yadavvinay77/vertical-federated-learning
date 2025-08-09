"""
Config module for VFL system.
Contains parameters for parties, dataset, encryption, and model.
"""

import yaml

DEFAULT_CONFIG = {
    "parties": ["client1", "client2"],
    "dataset": {
        "name": "adult",
        "data_path": "./data/adult_vertical_split.csv",
        "target_column": "income"
    },
    "encryption": {
        "use_real_he": False,
        "he_params": {
            "poly_mod_degree": 8192,
            "coeff_mod_bit_sizes": [60, 40, 40, 60],
            "global_scale": 2 ** 40
        }
    },
    "model": {
        "input_dims": [6, 6],  # Example split feature counts per client
        "learning_rate": 0.01,
        "epochs": 5,
        "batch_size": 32
    }
}

def load_config(path=None):
    if path:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG
