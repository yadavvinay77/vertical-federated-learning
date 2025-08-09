"""
Coordinator server logic.
Receives encrypted intermediate outputs from clients.
Aggregates encrypted data securely.
Decrypts aggregated intermediate to compute final output and loss.
Sends encrypted gradients back to clients.
"""

import numpy as np
from encryption.homomorphic import get_he_context
from models.model import ServerModel

class VFLServer:
    def __init__(self, config):
        self.config = config
        self.num_clients = len(config["parties"])
        self.input_dim = 10 * self.num_clients  # each client has 10 hidden dims
        self.model = ServerModel(self.input_dim)
        self.he = get_he_context(
            config["encryption"]["use_real_he"],
            config["encryption"]["he_params"]
        )

    def aggregate_encrypted(self, enc_intermediates):
        # Secure aggregation of encrypted vectors
        agg = enc_intermediates[0]
        for enc_vec in enc_intermediates[1:]:
            agg = self.he.add(agg, enc_vec)
        return agg

    def forward(self, aggregated_intermediate):
        # Decrypt aggregated intermediate
        intermediate = np.array(self.he.decrypt(aggregated_intermediate))
        # Forward server model
        return self.model.forward(intermediate)

    def backward(self, aggregated_intermediate, pred, labels):
        intermediate = np.array(self.he.decrypt(aggregated_intermediate))
        grad_w, grad_b = self.model.backward(intermediate, pred, labels)
        return grad_w, grad_b

    def update_model(self, grad_w, grad_b):
        lr = self.config["model"]["learning_rate"]
        self.model.update(grad_w, grad_b, lr)
