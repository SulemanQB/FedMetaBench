"""FedMeta-Env algorithms.

Implements:
- EnvMAML: MAML for station-level few-shot adaptation (regression)
- FedEnvMAML: Federated EnvMAML with station-as-client
- Baselines: FedAvg, Local-only, Fine-tune, Per-FedAvg
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EnvMAML:
    """MAML for environmental time-series — regression variant.

    Adapts to new station with K-day support set.
    Uses MSE loss for pollutant prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        """Adapt model to support set."""
        adapted = copy.deepcopy(self.model)
        adapted.train()

        for _ in range(self.inner_steps):
            pred = adapted(support_x)
            loss = F.mse_loss(pred, support_y)
            grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=True)
            for p, g in zip(adapted.parameters(), grads):
                p.data = p.data - self.inner_lr * g

        return adapted

    def meta_train_step(
        self,
        tasks: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        """One outer-loop step across tasks."""
        self.meta_optimizer.zero_grad()
        total_loss = 0.0

        for task in tasks:
            sx = task["support_x"].to(self.device)
            sy = task["support_y"].to(self.device)
            qx = task["query_x"].to(self.device)
            qy = task["query_y"].to(self.device)

            adapted = self.inner_loop(sx, sy)
            query_pred = adapted(qx)
            query_loss = F.mse_loss(query_pred, qy)
            total_loss += query_loss

        avg_loss = total_loss / len(tasks)
        avg_loss.backward()
        self.meta_optimizer.step()

        return {"meta_loss": avg_loss.item()}

    def evaluate(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> dict[str, float]:
        """Adapt on support, evaluate on query."""
        sx = support_x.to(self.device)
        sy = support_y.to(self.device)
        qx = query_x.to(self.device)
        qy = query_y.to(self.device)

        with torch.no_grad():
            pre_pred = self.model(qx)
            pre_mse = F.mse_loss(pre_pred, qy).item()

        # Adapt with gradient for inner loop
        adapted = copy.deepcopy(self.model)
        adapted.train()
        opt = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            opt.zero_grad()
            loss = F.mse_loss(adapted(sx), sy)
            loss.backward()
            opt.step()

        adapted.eval()
        with torch.no_grad():
            post_pred = adapted(qx)
            post_mse = F.mse_loss(post_pred, qy).item()
            mae = (post_pred - qy).abs().mean().item()

        return {
            "pre_adapt_mse": pre_mse,
            "post_adapt_mse": post_mse,
            "mae": mae,
            "rmse": post_mse ** 0.5,
            "improvement": (pre_mse - post_mse) / (pre_mse + 1e-8),
        }


class FedEnvMAML:
    """Federated EnvMAML — stations as clients.

    Server coordinates meta-learning across training stations,
    then evaluates few-shot adaptation on held-out stations.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        clients_per_round: int = 5,
        local_meta_steps: int = 3,
        device: str = "cpu",
    ):
        self.global_model = model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.clients_per_round = clients_per_round
        self.local_meta_steps = local_meta_steps
        self.device = device

    def train_round(
        self,
        fed_data,
        train_stations: list[int],
        k_days: int = 7,
    ) -> dict[str, float]:
        """One federated round with station-level meta-learning."""
        n_select = min(self.clients_per_round, len(train_stations))
        selected = np.random.choice(train_stations, n_select, replace=False).tolist()

        client_models = []
        client_losses = []

        for station_id in selected:
            local_model = copy.deepcopy(self.global_model)
            maml = EnvMAML(
                local_model, self.inner_lr, self.outer_lr,
                self.inner_steps, self.device,
            )

            # Create tasks from this station's data
            split = fed_data.get_k_shot_split(station_id, k_days)
            for _ in range(self.local_meta_steps):
                metrics = maml.meta_train_step([split])

            client_models.append(maml.model)
            client_losses.append(metrics["meta_loss"])

        # FedAvg aggregation
        self._aggregate(client_models)

        return {
            "avg_loss": float(np.mean(client_losses)),
            "n_clients": n_select,
        }

    def _aggregate(self, client_models: list[nn.Module]):
        n = len(client_models)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = sum(
                m.state_dict()[key].float() for m in client_models
            ) / n
        self.global_model.load_state_dict(global_dict)

    def evaluate_loso(
        self,
        fed_data,
        test_station: int,
        k_days_list: list[int] = None,
    ) -> dict[str, Any]:
        """Evaluate on held-out station with varying K-shot."""
        if k_days_list is None:
            k_days_list = [1, 3, 7, 14, 30]

        results = {}
        for k in k_days_list:
            split = fed_data.get_k_shot_split(test_station, k)
            maml = EnvMAML(
                copy.deepcopy(self.global_model),
                self.inner_lr, self.outer_lr,
                self.inner_steps, self.device,
            )
            eval_result = maml.evaluate(
                split["support_x"], split["support_y"],
                split["query_x"], split["query_y"],
            )
            results[f"k{k}"] = eval_result

        return results


class LocalOnly:
    """Baseline: train on single station only, no federation."""

    def __init__(self, model: nn.Module, lr: float = 0.01, device: str = "cpu"):
        self.model = model.to(device)
        self.lr = lr
        self.device = device

    def train_and_evaluate(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        epochs: int = 50,
    ) -> dict[str, float]:
        model = copy.deepcopy(self.model)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        sx = support_x.to(self.device)
        sy = support_y.to(self.device)

        for _ in range(epochs):
            opt.zero_grad()
            loss = F.mse_loss(model(sx), sy)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(query_x.to(self.device))
            qy = query_y.to(self.device)
            mse = F.mse_loss(pred, qy).item()
            mae = (pred - qy).abs().mean().item()

        return {"mse": mse, "rmse": mse ** 0.5, "mae": mae}


class FedAvgEnv:
    """Baseline: Standard FedAvg for regression, no meta-learning."""

    def __init__(self, model: nn.Module, lr: float = 0.01, local_epochs: int = 5, device: str = "cpu"):
        self.global_model = model.to(device)
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device

    def train_round(
        self,
        fed_data,
        train_stations: list[int],
        k_days: int = 7,
    ) -> dict[str, float]:
        client_models = []
        for sid in train_stations:
            local_model = copy.deepcopy(self.global_model)
            local_model.train()
            opt = torch.optim.SGD(local_model.parameters(), lr=self.lr)
            split = fed_data.get_k_shot_split(sid, k_days)
            sx = split["support_x"].to(self.device)
            sy = split["support_y"].to(self.device)

            for _ in range(self.local_epochs):
                opt.zero_grad()
                loss = F.mse_loss(local_model(sx), sy)
                loss.backward()
                opt.step()

            client_models.append(local_model)

        # Aggregate
        n = len(client_models)
        gd = self.global_model.state_dict()
        for key in gd:
            gd[key] = sum(m.state_dict()[key].float() for m in client_models) / n
        self.global_model.load_state_dict(gd)
        return {"n_clients": n}

    def evaluate(
        self,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> dict[str, float]:
        self.global_model.eval()
        with torch.no_grad():
            pred = self.global_model(query_x.to(self.device))
            qy = query_y.to(self.device)
            mse = F.mse_loss(pred, qy).item()
            mae = (pred - qy).abs().mean().item()
        return {"mse": mse, "rmse": mse ** 0.5, "mae": mae}
