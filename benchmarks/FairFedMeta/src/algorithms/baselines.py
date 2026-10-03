"""Baseline algorithms for comparison with FairFedMeta.

Implements:
- StandardFedAvg: vanilla FedAvg without fairness
- PerFedAvg: personalized FedAvg (MAML-based) without fairness
- FairFedAvg: FedAvg with equal group weights (no meta-learning)
- AgnosticFairFL: agnostic fairness approach (Mohri et al. style)
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardFedAvg:
    """Vanilla FedAvg baseline — no fairness consideration."""

    def __init__(self, model: nn.Module, lr: float = 0.01, local_epochs: int = 5):
        self.global_model = model
        self.lr = lr
        self.local_epochs = local_epochs

    def local_update(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        for _ in range(self.local_epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model(support_x), support_y)
            loss.backward()
            opt.step()
        return model

    def aggregate(self, client_models: list[nn.Module], weights: list[float] | None = None):
        if weights is None:
            weights = [1.0 / len(client_models)] * len(client_models)
        total = sum(weights)
        weights = [w / total for w in weights]

        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = sum(
                w * m.state_dict()[key].float() for w, m in zip(weights, client_models)
            )
        self.global_model.load_state_dict(global_dict)

    def train_round(
        self,
        client_data: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        client_models = []
        total_loss = 0.0

        for data in client_data:
            local_model = copy.deepcopy(self.global_model)
            local_model = self.local_update(
                local_model, data["support_x"], data["support_y"]
            )
            client_models.append(local_model)

            with torch.no_grad():
                loss = F.cross_entropy(
                    local_model(data["query_x"]), data["query_y"]
                )
                total_loss += loss.item()

        self.aggregate(client_models)
        return {"avg_loss": total_loss / len(client_data)}

    def evaluate(
        self,
        test_data: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data in test_data:
                preds = self.global_model(data["query_x"]).argmax(dim=1)
                correct += (preds == data["query_y"]).sum().item()
                total += len(data["query_y"])
        return {"accuracy": correct / total if total > 0 else 0.0}


class PerFedAvg:
    """First-order Reptile-style personalization baseline."""

    def __init__(
        self, model: nn.Module, inner_lr: float = 0.01,
        outer_lr: float = 0.001, inner_steps: int = 1,
    ):
        self.global_model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

    def local_meta_update(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        opt = torch.optim.SGD(local_model.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):
            opt.zero_grad()
            loss = F.cross_entropy(local_model(support_x), support_y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            query_loss = F.cross_entropy(local_model(query_x), query_y)
            preds = local_model(query_x).argmax(dim=1)
            acc = (preds == query_y).float().mean().item()

        return local_model.state_dict(), {"query_loss": query_loss.item(), "query_acc": acc}

    def train_round(
        self,
        client_data: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        total_acc = 0.0
        total_loss = 0.0
        # Accumulate Reptile deltas
        deltas = {k: torch.zeros_like(v) for k, v in self.global_model.state_dict().items()}

        for data in client_data:
            local_state, metrics = self.local_meta_update(
                data["support_x"], data["support_y"],
                data["query_x"], data["query_y"],
            )
            total_loss += metrics["query_loss"]
            total_acc += metrics["query_acc"]
            for k, v in self.global_model.state_dict().items():
                deltas[k] += local_state[k].float() - v.float()

        n = max(len(client_data), 1)
        with torch.no_grad():
            new_state = {
                k: v + self.outer_lr * (deltas[k] / n)
                for k, v in self.global_model.state_dict().items()
            }
            self.global_model.load_state_dict(new_state)

        return {
            "avg_loss": total_loss / n,
            "avg_acc": total_acc / n,
        }


class FairFedAvg:
    """FedAvg with fixed equal group weights — fairness without meta-learning."""

    def __init__(
        self, model: nn.Module, lr: float = 0.01,
        local_epochs: int = 5, n_groups: int = 5,
    ):
        self.global_model = model
        self.lr = lr
        self.local_epochs = local_epochs
        self.n_groups = n_groups
        # Fixed equal weights — no meta-learning of lambdas
        self.group_weights = torch.ones(n_groups) / n_groups

    def local_update(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> nn.Module:
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        weights = self.group_weights.to(support_x.device)

        for _ in range(self.local_epochs):
            opt.zero_grad()
            logits = model(support_x)

            # Group-weighted loss with fixed equal weights (must match support length)
            total_loss = torch.zeros((), device=support_x.device)
            n_active = 0
            for g in range(self.n_groups):
                mask = group_ids == g
                if mask.any():
                    g_loss = F.cross_entropy(logits[mask], support_y[mask])
                    total_loss = total_loss + weights[g] * g_loss
                    n_active += 1
            if n_active == 0:
                total_loss = F.cross_entropy(logits, support_y)

            total_loss.backward()
            opt.step()
        return model

    def train_round(
        self,
        client_data: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        client_models = []
        for data in client_data:
            local_model = copy.deepcopy(self.global_model)
            # Prefer support_group_ids so masks align with support_x
            gids = data.get("support_group_ids", data["group_ids"])
            local_model = self.local_update(
                local_model, data["support_x"], data["support_y"], gids,
            )
            client_models.append(local_model)

        # FedAvg aggregation
        global_dict = self.global_model.state_dict()
        n = len(client_models)
        for key in global_dict:
            global_dict[key] = sum(m.state_dict()[key].float() for m in client_models) / n
        self.global_model.load_state_dict(global_dict)

        return {"n_clients": n}


class AgnosticFairFL:
    """Reweight clients according to their worst-performing group."""

    def __init__(
        self, model: nn.Module, lr: float = 0.01,
        local_epochs: int = 5, n_groups: int = 5, lambda_fair: float = 0.5,
    ):
        self.global_model = model
        self.lr = lr
        self.local_epochs = local_epochs
        self.n_groups = n_groups
        self.lambda_fair = lambda_fair  # mixing between avg and worst-group

    def local_update(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        for _ in range(self.local_epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model(support_x), support_y)
            loss.backward()
            opt.step()
        return model

    def compute_group_losses(
        self,
        model: nn.Module,
        data: dict[str, torch.Tensor],
    ) -> dict[int, float]:
        model.eval()
        group_losses = {}
        with torch.no_grad():
            logits = model(data["query_x"])
            for g in range(self.n_groups):
                mask = data["group_ids"] == g
                if mask.sum() > 0:
                    g_loss = F.cross_entropy(logits[mask], data["query_y"][mask])
                    group_losses[g] = g_loss.item()
        return group_losses

    def train_round(
        self,
        client_data: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        client_models = []
        client_group_losses = []

        for data in client_data:
            local_model = copy.deepcopy(self.global_model)
            local_model = self.local_update(
                local_model, data["support_x"], data["support_y"]
            )
            client_models.append(local_model)
            group_losses = self.compute_group_losses(local_model, data)
            client_group_losses.append(group_losses)

        # Compute client weights based on worst-group loss
        client_weights = []
        for gl in client_group_losses:
            if gl:
                worst = max(gl.values())
                avg = sum(gl.values()) / len(gl)
                # Higher weight to clients with worse worst-group performance
                w = self.lambda_fair * worst + (1 - self.lambda_fair) * avg
            else:
                w = 1.0
            client_weights.append(w)

        total_w = sum(client_weights)
        client_weights = [w / total_w for w in client_weights]

        # Weighted aggregation
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = sum(
                w * m.state_dict()[key].float()
                for w, m in zip(client_weights, client_models)
            )
        self.global_model.load_state_dict(global_dict)

        return {"client_weights": client_weights}
