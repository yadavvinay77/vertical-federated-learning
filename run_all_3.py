import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== Models ======

def stable_sigmoid(z):
    z = np.clip(z, -500, 500)
    out = np.empty_like(z)
    pos_mask = (z >= 0)
    neg_mask = ~pos_mask
    out[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[neg_mask])
    out[neg_mask] = exp_z / (1 + exp_z)
    return out

class ClientModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        # smaller init scale to avoid overflow
        self.weights = np.random.randn(input_dim, 10) * 1e-4  
        self.bias = np.zeros(10)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

    def update(self, grad_w, grad_b, lr):
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b

class ServerModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.random.randn(input_dim, 1) * 1e-4
        self.bias = 0.0

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return stable_sigmoid(z)

    def compute_loss(self, pred, labels):
        epsilon = 1e-7
        pred = np.clip(pred, epsilon, 1 - epsilon)
        return -np.mean(labels * np.log(pred) + (1 - labels) * np.log(1 - pred))

    def backward(self, x, pred, labels):
        m = labels.shape[0]
        grad_pred = pred - labels.reshape(-1,1)
        grad_w = np.dot(x.T, grad_pred) / m
        grad_b = np.mean(grad_pred)
        grad_intermediate = np.dot(grad_pred, self.weights.T)
        return grad_w, grad_b, grad_intermediate

    def update(self, grad_w, grad_b, lr):
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b

# ====== Data Loading & Scaling ======

def load_clients_data(num_clients):
    clients_data = []
    for i in range(1, num_clients + 1):
        filename = f"data/client{i}_data.csv"
        df = pd.read_csv(filename)
        data = df.values.astype(np.float64)

        # Normalize each feature (mean=0, std=1) for numerical stability
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + 1e-8  # avoid div by zero
        data = (data - mean) / std

        clients_data.append(data)
    labels = pd.read_csv("data/labels.csv", header=None).values.flatten()

    # Check labels validity
    if not set(np.unique(labels)).issubset({0,1}):
        raise ValueError("Labels must be binary 0 or 1 only.")

    return clients_data, labels

# ====== Training and evaluation ======

def train_vertical_federated(clients_data, labels, epochs=10, batch_size=64, lr=0.01):
    n_samples = labels.shape[0]
    n_clients = len(clients_data)

    client_models = [ClientModel(data.shape[1]) for data in clients_data]
    server_model = ServerModel(10 * n_clients)

    losses = []

    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]

            client_outputs = []
            for i, model in enumerate(client_models):
                out = model.forward(clients_data[i][batch_idx])
                client_outputs.append(out)

            concat = np.hstack(client_outputs)

            preds = server_model.forward(concat)
            batch_labels = labels[batch_idx].reshape(-1,1)
            loss = server_model.compute_loss(preds, batch_labels)
            epoch_loss += loss * (end - start)

            grad_w, grad_b, grad_intermediate = server_model.backward(concat, preds, batch_labels)
            server_model.update(grad_w, grad_b, lr)

            for i, model in enumerate(client_models):
                grad_c = grad_intermediate[:, i*10:(i+1)*10]
                weights_grad = np.dot(clients_data[i][batch_idx].T, grad_c) / grad_c.shape[0]
                bias_grad = np.mean(grad_c, axis=0)
                model.update(weights_grad, bias_grad, lr)

        avg_loss = epoch_loss / n_samples
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        if np.isnan(avg_loss) or np.isinf(avg_loss):
            print("Warning: loss became NaN or Inf, stopping training.")
            break

    return client_models, server_model, losses

def evaluate(client_models, server_model, clients_data, labels):
    client_outputs = []
    for i, model in enumerate(client_models):
        out = model.forward(clients_data[i])
        client_outputs.append(out)
    concat = np.hstack(client_outputs)
    preds = server_model.forward(concat).flatten()
    pred_labels = (preds >= 0.5).astype(int)
    accuracy = np.mean(pred_labels == labels)
    loss = server_model.compute_loss(preds.reshape(-1,1), labels.reshape(-1,1))
    print(f"Evaluation Loss: {loss:.6f}, Accuracy: {accuracy*100:.2f}%")

def save_models(client_models, server_model, prefix="vfl_model"):
    for i, model in enumerate(client_models, 1):
        np.savez(f"{prefix}_client{i}.npz", weights=model.weights, bias=model.bias)
    np.savez(f"{prefix}_server.npz", weights=server_model.weights, bias=server_model.bias)
    print(f"Models saved with prefix '{prefix}'")

def plot_losses(losses):
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()

# ====== Main ======

def main():
    num_clients = 2  # Change as needed

    clients_data, labels = load_clients_data(num_clients)

    client_models, server_model, losses = train_vertical_federated(
        clients_data, labels, epochs=10, batch_size=64, lr=0.01)

    evaluate(client_models, server_model, clients_data, labels)
    save_models(client_models, server_model)
    plot_losses(losses)

if __name__ == "__main__":
    main()
