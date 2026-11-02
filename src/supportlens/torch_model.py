"""A small CPU network; sparse TF-IDF is densified only one batch at a time."""
import os
import numpy as np
import torch
from torch import nn


class IntentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.layers(x)  # raw logits; CrossEntropyLoss applies log-softmax internally


def train_network(x, y, *, hidden_dim, epochs, batch_size, learning_rate, seed, n_classes):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = IntentNet(x.shape[1], hidden_dim, n_classes).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for ids in np.array_split(rng.permutation(x.shape[0]), np.ceil(x.shape[0] / batch_size).astype(int)):
            inputs = torch.from_numpy(x[ids].toarray().astype(np.float32, copy=False))
            targets = torch.as_tensor(y[ids], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(ids)
        losses.append(total / x.shape[0])
        print(f"epoch {epoch + 1}/{epochs}: training loss={losses[-1]:.4f}", flush=True)
    return model, losses


def predict_probabilities(model, x, batch_size=128):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            inputs = torch.from_numpy(x[start:start + batch_size].toarray().astype(np.float32, copy=False))
            outputs.append(torch.softmax(model(inputs), dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)
