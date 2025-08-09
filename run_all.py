import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from colorama import Fore, Style, init
from models.model import ClientModel, ServerModel

init(autoreset=True)

def load_config():
    return {
        "dataset": {
            "client1_path": "data/client1_data.csv",
            "client2_path": "data/client2_data.csv",
            "labels_path": "data/labels.csv"
        },
        "training": {
            "epochs": 3,
            "batch_size": 64,
            "learning_rate": 0.01
        }
    }

class ClientDataset:
    def __init__(self, path):
        df = pd.read_csv(path)
        self.features = df.values.astype(np.float64)

def client_process(client_id, data, conn, epochs, batch_size, lr):
    print(f"{Fore.CYAN}[Client {client_id}]{Style.RESET_ALL} Starting training with {len(data)} samples")
    input_dim = data.shape[1]
    model = ClientModel(input_dim)

    for epoch in range(epochs):
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        batches = (len(data) + batch_size - 1) // batch_size
        try:
            with tqdm(total=batches, desc=f"Client {client_id} training", leave=False, position=client_id+1) as pbar:
                for start in range(0, len(data), batch_size):
                    end = min(start + batch_size, len(data))
                    batch_idx = indices[start:end]
                    x_batch = data[batch_idx]

                    # Forward pass on client
                    intermediate = model.forward(x_batch)

                    # Send intermediate to server
                    conn.send(intermediate)

                    # Receive gradient w.r.t intermediate features and bias grad from server
                    grad_intermediate, grad_bias = conn.recv()  # shapes: (batch_size, 10), scalar or vector

                    # Compute gradients w.r.t client weights and bias
                    weights_grad = np.dot(x_batch.T, grad_intermediate) / x_batch.shape[0]  # average over batch
                    bias_grad = np.mean(grad_intermediate, axis=0)

                    # Update client model parameters
                    model.update(weights_grad, bias_grad, lr)
                    pbar.update(1)
        finally:
            pass  # ensures tqdm bar closes properly on interruption

        print(f"{Fore.CYAN}[Client {client_id}]{Style.RESET_ALL} Epoch {epoch+1}/{epochs} done")

    print(f"{Fore.CYAN}[Client {client_id}]{Style.RESET_ALL} Training complete")
    conn.send("DONE")

def server_process(conns, labels, epochs, batch_size, lr):
    print(f"{Fore.GREEN}[Server]{Style.RESET_ALL} Starting training")
    n_clients = len(conns)
    hidden_dim = 10 * n_clients  # each client outputs 10 features
    model = ServerModel(hidden_dim)

    for epoch in range(epochs):
        batches = (len(labels) + batch_size - 1) // batch_size
        try:
            with tqdm(total=batches, desc="Server training", leave=False, position=0) as pbar:
                for start in range(0, len(labels), batch_size):
                    end = min(start + batch_size, len(labels))
                    batch_labels = labels[start:end].reshape(-1, 1)

                    # Collect client intermediate outputs
                    intermediates = [conn.recv() for conn in conns]

                    # Concatenate client outputs
                    intermediate_concat = np.hstack(intermediates)

                    # Forward pass on server
                    preds = model.forward(intermediate_concat)

                    # Compute loss (for logging)
                    loss = model.compute_loss(preds, batch_labels)

                    # Backward pass on server
                    grad_w, grad_b, grad_intermediate = model.backward(intermediate_concat, preds, batch_labels)

                    # Update server model parameters
                    model.update(grad_w, grad_b, lr)

                    # Split gradient for each client along feature dimension (axis=1)
                    grad_per_client = np.split(grad_intermediate, n_clients, axis=1)

                    # Send gradients to clients
                    for i, conn in enumerate(conns):
                        conn.send((grad_per_client[i], grad_b))

                    pbar.set_postfix({"loss": f"{loss:.4f}"})
                    pbar.update(1)
        finally:
            pass  # ensures tqdm bar closes properly

        print(f"{Fore.GREEN}[Server]{Style.RESET_ALL} Epoch {epoch+1}/{epochs} done, Loss: {loss:.4f}")

    print(f"{Fore.GREEN}[Server]{Style.RESET_ALL} Training complete")

    for conn in conns:
        msg = conn.recv()
        if msg != "DONE":
            print(f"{Fore.RED}[Server]{Style.RESET_ALL} Unexpected message from client: {msg}")

def main():
    """
    Load datasets, start client processes and server training loop.
    """
    config = load_config()
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]

    client1_dataset = ClientDataset(config["dataset"]["client1_path"])
    client2_dataset = ClientDataset(config["dataset"]["client2_path"])
    labels = pd.read_csv(config["dataset"]["labels_path"], header=None).values.flatten()

    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(2)])

    clients = [
        mp.Process(target=client_process, args=(0, client1_dataset.features, child_conns[0], epochs, batch_size, lr)),
        mp.Process(target=client_process, args=(1, client2_dataset.features, child_conns[1], epochs, batch_size, lr)),
    ]

    for c in clients:
        c.start()

    server_process(parent_conns, labels, epochs, batch_size, lr)

    for c in clients:
        c.join()

if __name__ == "__main__":
    main()
