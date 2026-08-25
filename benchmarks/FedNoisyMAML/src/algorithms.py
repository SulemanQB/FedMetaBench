"""
FedMAML and FedAvg implementations for noisy-label comparison.
"""

from collections import OrderedDict
import copy
import torch
import torch.nn.functional as F


# ---------- Functional MAML helpers ----------

def functional_forward(model, params, x):
    """Forward pass using given params, supporting MLP and CNN architectures."""
    h = x
    keys = list(params.keys())
    idx = 0
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            h = F.linear(h, params[keys[idx]], params[keys[idx + 1]])
            idx += 2
        elif isinstance(layer, torch.nn.Conv2d):
            h = F.conv2d(h, params[keys[idx]], params[keys[idx + 1]],
                        stride=layer.stride, padding=layer.padding)
            idx += 2
        elif isinstance(layer, (torch.nn.ReLU, torch.nn.Tanh, torch.nn.Sigmoid)):
            h = layer(h)
        elif isinstance(layer, torch.nn.MaxPool2d):
            h = F.max_pool2d(h, layer.kernel_size, stride=layer.stride,
                            padding=layer.padding)
        elif isinstance(layer, torch.nn.Flatten):
            h = h.view(h.size(0), -1)
    return h


def maml_inner_adapt(model, x_support, y_support, inner_lr=0.01, inner_steps=5):
    """MAML inner-loop adaptation (functional, differentiable)."""
    params = OrderedDict(
        (name, p.clone()) for name, p in model.named_parameters()
    )

    for _ in range(inner_steps):
        logits = functional_forward(model, params, x_support)
        loss = F.cross_entropy(logits, y_support)
        grads = torch.autograd.grad(loss, list(params.values()), create_graph=True)
        params = OrderedDict(
            (name, p - inner_lr * g)
            for (name, p), g in zip(params.items(), grads)
        )

    return params


# ---------- FedMAML ----------

class FedMAML:
    """Federated MAML for classification."""

    def __init__(self, model, clients, outer_lr=0.001, inner_lr=0.01,
                 inner_steps=5):
        self.model = model
        self.clients = clients
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)

    def train_round(self, n_support=10, n_query=10):
        """One round of FedMAML across all clients."""
        self.optimizer.zero_grad()
        total_loss = 0.0
        device = next(self.model.parameters()).device

        for client in self.clients:
            xs, ys, xq, yq = client.sample_task_all_noisy(n_support, n_query)
            xs, ys = xs.to(device), ys.to(device)
            xq, yq = xq.to(device), yq.to(device)
            adapted_params = maml_inner_adapt(
                self.model, xs, ys, self.inner_lr, self.inner_steps)

            logits = functional_forward(self.model, adapted_params, xq)
            query_loss = F.cross_entropy(logits, yq)
            total_loss += query_loss.item()

            meta_grads = torch.autograd.grad(query_loss, self.model.parameters())
            with torch.no_grad():
                for p, g in zip(self.model.parameters(), meta_grads):
                    if p.grad is None:
                        p.grad = g / len(self.clients)
                    else:
                        p.grad += g / len(self.clients)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        return total_loss / len(self.clients)

    def evaluate(self, n_test=50):
        """Evaluate per-client accuracy after adaptation on clean test data."""
        accuracies = []
        device = next(self.model.parameters()).device
        for client in self.clients:
            xs, ys, _, _ = client.sample_task_all_noisy(10, 1)
            xs, ys = xs.to(device), ys.to(device)

            # Non-differentiable inner adaptation for eval
            adapted_params = OrderedDict(
                (name, p.clone().detach().requires_grad_(True))
                for name, p in self.model.named_parameters()
            )
            for _ in range(self.inner_steps):
                logits = functional_forward(self.model, adapted_params, xs)
                loss = F.cross_entropy(logits, ys)
                grads = torch.autograd.grad(loss, list(adapted_params.values()))
                adapted_params = OrderedDict(
                    (name, (p - self.inner_lr * g).detach().requires_grad_(True))
                    for (name, p), g in zip(adapted_params.items(), grads)
                )

            x_test, y_test = client.get_clean_test(n_test)
            x_test, y_test = x_test.to(device), y_test.to(device)
            with torch.no_grad():
                logits = functional_forward(self.model, adapted_params, x_test)
                preds = logits.argmax(dim=1)
                acc = (preds == y_test).float().mean().item()
            accuracies.append(acc)

        return accuracies


# ---------- FedAvg ----------

class FedAvg:
    """Standard FedAvg for classification."""

    def __init__(self, model, clients, lr=0.01, local_epochs=5):
        self.model = model
        self.clients = clients
        self.lr = lr
        self.local_epochs = local_epochs

    def train_round(self, n_samples=20):
        """One round of FedAvg."""
        global_state = copy.deepcopy(self.model.state_dict())
        client_states = []
        total_loss = 0.0
        device = next(self.model.parameters()).device

        for client in self.clients:
            # Local training
            local_model = copy.deepcopy(self.model)
            opt = torch.optim.SGD(local_model.parameters(), lr=self.lr)

            for _ in range(self.local_epochs):
                perm = torch.randperm(len(client.X))[:n_samples]
                x_batch = client.X[perm].to(device)
                y_batch = client.y_noisy[perm].to(device)
                logits = local_model(x_batch)
                loss = F.cross_entropy(logits, y_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            client_states.append(copy.deepcopy(local_model.state_dict()))

        # Average
        avg_state = OrderedDict()
        for key in global_state:
            avg_state[key] = torch.stack([s[key].float() for s in client_states]).mean(0)
        self.model.load_state_dict(avg_state)

        return total_loss / (len(self.clients) * self.local_epochs)

    def evaluate(self, n_test=50):
        """Evaluate per-client accuracy on clean test data."""
        accuracies = []
        device = next(self.model.parameters()).device
        for client in self.clients:
            x_test, y_test = client.get_clean_test(n_test)
            x_test, y_test = x_test.to(device), y_test.to(device)
            with torch.no_grad():
                logits = self.model(x_test)
                preds = logits.argmax(dim=1)
                acc = (preds == y_test).float().mean().item()
            accuracies.append(acc)
        return accuracies
