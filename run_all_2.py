import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== Models ======

def stable_sigmoid(z):
    z = np.clip(z, -500, 500)  # clip to avoid overflow
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
        self.weights = np.random.randn(input_dim, 10) * 0.001
        self.bias = np.zeros(10)  # bias vector

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

    def update(self, grad_w, grad_b, lr):
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b

class ServerModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.random.randn(input_dim, 1) * 0.001
        self.bias = 0.0  # scalar bias

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

# ====== Data Loading ======

def load_data():
    client1 = pd.read_csv("data/client1_data.csv").values.astype(np.float64)
    client2 = pd.read_csv("data/client2_data.csv").values.astype(np.float64)
    labels = pd.read_csv("data/labels.csv", header=None).values.flatten()
    
    # Standardize features for each client
    client1 = (client1 - client1.mean(axis=0)) / (client1.std(axis=0) + 1e-8)
    client2 = (client2 - client2.mean(axis=0)) / (client2.std(axis=0) + 1e-8)
    
    return client1, client2, labels

# ====== Training and Evaluation ======

def train_vertical_federated(client1_data, client2_data, labels, epochs=3, batch_size=64, lr=0.001):
    n_samples = labels.shape[0]
    client1_model = ClientModel(client1_data.shape[1])
    client2_model = ClientModel(client2_data.shape[1])
    server_model = ServerModel(20)  # 10 features from each client => 20 total

    losses = []

    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]

            c1_out = client1_model.forward(client1_data[batch_idx])
            c2_out = client2_model.forward(client2_data[batch_idx])

            concat = np.hstack([c1_out, c2_out])
            preds = server_model.forward(concat)

            batch_labels = labels[batch_idx].reshape(-1,1)
            loss = server_model.compute_loss(preds, batch_labels)
            epoch_loss += loss * (end - start)

            grad_w, grad_b, grad_intermediate = server_model.backward(concat, preds, batch_labels)
            server_model.update(grad_w, grad_b, lr)

            grad_c1 = grad_intermediate[:,:10]
            grad_c2 = grad_intermediate[:,10:]

            weights_grad_c1 = np.dot(client1_data[batch_idx].T, grad_c1) / grad_c1.shape[0]
            bias_grad_c1 = np.mean(grad_c1, axis=0)
            client1_model.update(weights_grad_c1, bias_grad_c1, lr)

            weights_grad_c2 = np.dot(client2_data[batch_idx].T, grad_c2) / grad_c2.shape[0]
            bias_grad_c2 = np.mean(grad_c2, axis=0)
            client2_model.update(weights_grad_c2, bias_grad_c2, lr)

            if np.isnan(loss) or np.isinf(loss):
                print("Warning: loss became NaN or Inf, stopping training.")
                return client1_model, client2_model, server_model, losses

        avg_loss = epoch_loss / n_samples
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return client1_model, client2_model, server_model, losses

def evaluate(client1_model, client2_model, server_model, client1_data, client2_data, labels):
    c1_out = client1_model.forward(client1_data)
    c2_out = client2_model.forward(client2_data)
    concat = np.hstack([c1_out, c2_out])
    preds = server_model.forward(concat).flatten()
    pred_labels = (preds >= 0.5).astype(int)
    accuracy = np.mean(pred_labels == labels)
    loss = server_model.compute_loss(preds.reshape(-1,1), labels.reshape(-1,1))
    print(f"Evaluation Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")

def save_models(client1_model, client2_model, server_model, prefix="vfl_model"):
    np.savez(prefix + "_client1.npz", weights=client1_model.weights, bias=client1_model.bias)
    np.savez(prefix + "_client2.npz", weights=client2_model.weights, bias=client2_model.bias)
    np.savez(prefix + "_server.npz", weights=server_model.weights, bias=server_model.bias)
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
    client1_data, client2_data, labels = load_data()
    client1_model, client2_model, server_model, losses = train_vertical_federated(client1_data, client2_data, labels, epochs=10, batch_size=64, lr=0.001)
    evaluate(client1_model, client2_model, server_model, client1_data, client2_data, labels)
    save_models(client1_model, client2_model, server_model)
    plot_losses(losses)

if __name__ == "__main__":
    main()
