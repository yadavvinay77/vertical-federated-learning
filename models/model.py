import numpy as np

def stable_sigmoid(z):
    out = np.empty_like(z)
    pos_mask = (z >= 0)
    neg_mask = ~pos_mask
    out[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[neg_mask])
    out[neg_mask] = exp_z / (1 + exp_z)
    return out

def clip_gradient(grad, max_norm=1.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad

class ClientModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.random.randn(input_dim, 10) * 0.001  # smaller init scale
        self.bias = np.zeros(10)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

    def update(self, grad_w, grad_b, lr):
        grad_w = clip_gradient(grad_w)
        grad_b = clip_gradient(grad_b)
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b

class ServerModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.random.randn(input_dim, 1) * 0.001  # smaller init scale
        self.bias = 0.0

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        z = np.clip(z, -500, 500)  # clip logits to avoid overflow in exp
        return stable_sigmoid(z)

    def compute_loss(self, pred, labels):
        epsilon = 1e-7
        pred = np.clip(pred, epsilon, 1 - epsilon)
        return -np.mean(labels * np.log(pred) + (1 - labels) * np.log(1 - pred))

    def backward(self, x, pred, labels):
        m = labels.shape[0]
        grad_pred = pred - labels.reshape(-1, 1)
        grad_w = np.dot(x.T, grad_pred) / m
        grad_b = np.mean(grad_pred)
        grad_intermediate = np.dot(grad_pred, self.weights.T)
        return grad_w, grad_b, grad_intermediate

    def update(self, grad_w, grad_b, lr):
        grad_w = clip_gradient(grad_w)
        grad_b = clip_gradient(np.array([grad_b]))
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b[0]
