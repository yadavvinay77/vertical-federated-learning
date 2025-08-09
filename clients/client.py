"""
Client-side logic of VFL.
Loads local data partition and model.
Encrypts intermediate output and sends to server.
Receives aggregated encrypted output and computes loss/gradients.
"""

import numpy as np
from encryption.homomorphic import get_he_context
from models.model import ClientModel

class VFLClient:
    def __init__(self, client_id, dataset, config):
        self.client_id = client_id
        self.features = dataset.features
        self.labels = dataset.labels
        self.config = config
        # initialize model etc.

    def get_batch(self, start, end):
        return self.features[start:end]

    def forward(self, x):
        # Forward pass local model
        x = np.array(x, dtype=np.float64)
        return self.model.forward(x)

    def encrypt_intermediate(self, intermediate):
        # Encrypt intermediate representation before sending
        return self.he.encrypt(intermediate.tolist())

    def decrypt_gradient(self, enc_grad):
        # Decrypt aggregated gradient received from server
        return np.array(self.he.decrypt(enc_grad))

    def update_model(self, grad_w, grad_b):
        lr = self.config["model"]["learning_rate"]
        self.model.update(grad_w, grad_b, lr)

    def train_batch(self, x_batch, enc_agg_grad):
        # Forward local model
        intermediate = self.forward(x_batch)
        # Encrypt intermediate to send to server
        encrypted_intermediate = self.encrypt_intermediate(intermediate)
        # Here we simulate sending and receiving in same process for demo
        decrypted_grad = self.decrypt_gradient(enc_agg_grad)
        # Update client model with gradient
        grad_w = np.dot(x_batch.T, decrypted_grad) / x_batch.shape[0]
        grad_b = np.mean(decrypted_grad, axis=0)
        self.update_model(grad_w, grad_b)
        return encrypted_intermediate
