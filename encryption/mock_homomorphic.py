import numpy as np

def encrypt(data):
    if isinstance(data, dict):
        # For gradients dict: encrypt each numpy array/float individually
        return {k: np.array(v, dtype=float) if not isinstance(v, dict) else encrypt(v) for k, v in data.items()}
    else:
        # For intermediate outputs: wrap numpy array in dict with key "encrypted"
        return {"encrypted": np.array(data, dtype=float)}

def decrypt(enc_data):
    if isinstance(enc_data, dict):
        # Detect if this dict is a gradients dict (keys like 'weights_grad', 'bias_grad')
        # or intermediate output dict (single key "encrypted")
        if set(enc_data.keys()) == {"encrypted"}:
            return enc_data["encrypted"]
        elif all(isinstance(v, (np.ndarray, float, int)) for v in enc_data.values()):
            return {k: v for k, v in enc_data.items()}
        else:
            return {k: decrypt(v) for k, v in enc_data.items()}
    else:
        raise ValueError("Invalid encrypted data format")
