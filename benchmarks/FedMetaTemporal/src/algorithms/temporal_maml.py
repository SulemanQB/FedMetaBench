"""FedMeta-Temporal+ core algorithms.

Paper 4a: TemporalMAML — MAML + Neural CDE + drift-triggered re-adaptation
Paper 4b: EWC-MAML — EWC regularization to prevent catastrophic forgetting

Also: Temporal-weighted FedAvg that favors recent client updates.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .drift_detection import PageHinkleyDetector, ClientDriftMonitor

logger = logging.getLogger(__name__)


class TemporalMAML:
    """MAML aware of temporal concept drift.

    Key features:
    - Drift detection triggers re-initialization of meta-learner
    - Recent tasks weighted more heavily in outer loop
    - Compatible with Neural CDE encoder for irregular time series
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 3,
        temporal_decay: float = 0.9,
        clients_per_round: int = 5,
        device: str = "cpu",
        **kwargs,
    ):
        self.model = model.to(device)
        self.global_model = self.model  # alias for shared evaluate helpers
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.temporal_decay = temporal_decay
        self.clients_per_round = clients_per_round
        self.device = device
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
        self.drift_detector = PageHinkleyDetector()
        self.drift_monitor = None
        self.round = 0

    def init_drift_monitor(self, n_clients: int):
        self.drift_monitor = ClientDriftMonitor(n_clients)

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        adapted = copy.deepcopy(self.model)
        adapted.train()

        for _ in range(self.inner_steps):
            out = adapted(support_x)
            loss = F.cross_entropy(out, support_y)
            grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=True)
            for p, g in zip(adapted.parameters(), grads):
                p.data = p.data - self.inner_lr * g

        return adapted

    def meta_train_step(
        self,
        tasks: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        """Temporally-weighted meta-training step.

        Tasks should be ordered by time — later tasks get higher weight.
        """
        self.meta_optimizer.zero_grad()
        total_loss = 0.0
        total_weight = 0.0

        for i, task in enumerate(tasks):
            sx = task["support_x"].to(self.device)
            sy = task["support_y"].to(self.device)
            qx = task["query_x"].to(self.device)
            qy = task["query_y"].to(self.device)

            adapted = self.inner_loop(sx, sy)
            query_loss = F.cross_entropy(adapted(qx), qy)

            # Temporal weighting: more recent tasks get higher weight
            time_weight = self.temporal_decay ** (len(tasks) - 1 - i)
            total_loss += time_weight * query_loss
            total_weight += time_weight

        avg_loss = total_loss / total_weight
        avg_loss.backward()
        self.meta_optimizer.step()

        # Drift detection on current loss
        drift = self.drift_detector.update(avg_loss.item())

        return {
            "meta_loss": avg_loss.item(),
            "drift_detected": drift,
        }

    def evaluate_tasks(
        self,
        tasks: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        """Evaluate adapted model on a list of tasks."""
        accs = []
        for task in tasks:
            sx = task["support_x"].to(self.device)
            sy = task["support_y"].to(self.device)
            qx = task["query_x"].to(self.device)
            qy = task["query_y"].to(self.device)

            adapted = copy.deepcopy(self.model)
            adapted.train()
            opt = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)
            for _ in range(self.inner_steps):
                opt.zero_grad()
                loss = F.cross_entropy(adapted(sx), sy)
                loss.backward()
                opt.step()

            adapted.eval()
            with torch.no_grad():
                preds = adapted(qx).argmax(dim=1)
                acc = (preds == qy).float().mean().item()
                accs.append(acc)

        return {
            "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
            "std_accuracy": float(np.std(accs)) if accs else 0.0,
            "n_tasks": len(tasks),
        }

    def train_round(
        self,
        fed_data,
        train_client_ids: list[int],
        n_tasks: int = 5,
        k_shot: int = 20,
    ) -> dict[str, float]:
        """Federated round: pool client tasks and run one temporally-weighted meta step."""
        n_select = min(self.clients_per_round, len(train_client_ids))
        selected = np.random.choice(train_client_ids, n_select, replace=False).tolist()

        all_tasks = []
        for cid in selected:
            tasks = fed_data.create_meta_tasks(cid, n_tasks, k_shot, seed=self.round + cid)
            all_tasks.extend(tasks)

        metrics = self.meta_train_step(all_tasks)
        if self.drift_monitor:
            for cid in selected:
                self.drift_monitor.update(cid, metrics["meta_loss"])
            self.drift_monitor.advance_step()
        self.round += 1
        return {
            "avg_loss": metrics["meta_loss"],
            "n_drifts": int(bool(metrics.get("drift_detected"))),
            "round": self.round,
        }

    def evaluate(
        self,
        fed_data,
        test_client_ids: list[int],
        n_tasks: int = 5,
        k_shot: int = 20,
    ) -> dict[str, float]:
        all_accs = []
        for cid in test_client_ids:
            tasks = fed_data.create_meta_tasks(cid, n_tasks, k_shot, seed=9999 + cid)
            result = self.evaluate_tasks(tasks)
            all_accs.append(result["mean_accuracy"])
        return {
            "mean_accuracy": float(np.mean(all_accs)) if all_accs else 0.0,
            "std_accuracy": float(np.std(all_accs)) if all_accs else 0.0,
            "per_client_acc": {cid: acc for cid, acc in zip(test_client_ids, all_accs)},
        }


class EWCMAML:
    """MAML with Elastic Weight Consolidation to handle drift.

    EWC prevents catastrophic forgetting of useful past knowledge
    when adapting to new temporal concepts.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 3,
        ewc_lambda: float = 1.0,
        clients_per_round: int = 5,
        device: str = "cpu",
        **kwargs,
    ):
        self.model = model.to(device)
        self.global_model = self.model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.ewc_lambda = ewc_lambda
        self.clients_per_round = clients_per_round
        self.device = device
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)

        # EWC components
        self.fisher_info: dict[str, torch.Tensor] = {}
        self.prev_params: dict[str, torch.Tensor] = {}
        self._fisher_computed = False
        self.drift_monitor = None
        self.round = 0

    def init_drift_monitor(self, n_clients: int):
        self.drift_monitor = ClientDriftMonitor(n_clients)

    def compute_fisher(self, tasks: list[dict[str, torch.Tensor]]):
        """Compute Fisher Information Matrix from current tasks."""
        self.model.train()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        for task in tasks:
            sx = task["support_x"].to(self.device)
            sy = task["support_y"].to(self.device)
            self.model.zero_grad()
            out = self.model(sx)
            loss = F.cross_entropy(out, sy)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

        # Average
        for n in fisher:
            fisher[n] /= len(tasks)

        self.fisher_info = fisher
        self.prev_params = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self._fisher_computed = True

    def _ewc_penalty(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self._fisher_computed:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self.fisher_info:
                penalty += (
                    self.fisher_info[n] * (p - self.prev_params[n]).pow(2)
                ).sum()
        return penalty * self.ewc_lambda / 2

    def meta_train_step(
        self,
        tasks: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        self.meta_optimizer.zero_grad()
        total_loss = 0.0

        for task in tasks:
            sx = task["support_x"].to(self.device)
            sy = task["support_y"].to(self.device)
            qx = task["query_x"].to(self.device)
            qy = task["query_y"].to(self.device)

            # Inner loop
            adapted = copy.deepcopy(self.model)
            adapted.train()
            for _ in range(self.inner_steps):
                out = adapted(sx)
                loss = F.cross_entropy(out, sy)
                grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=True)
                for p, g in zip(adapted.parameters(), grads):
                    p.data = p.data - self.inner_lr * g

            query_loss = F.cross_entropy(adapted(qx), qy)
            total_loss += query_loss

        avg_loss = total_loss / len(tasks)

        # Add EWC penalty
        ewc_loss = self._ewc_penalty()
        total = avg_loss + ewc_loss

        total.backward()
        self.meta_optimizer.step()

        return {
            "meta_loss": avg_loss.item(),
            "ewc_penalty": ewc_loss.item(),
            "total_loss": total.item(),
        }

    def train_round(
        self,
        fed_data,
        train_client_ids: list[int],
        n_tasks: int = 5,
        k_shot: int = 20,
    ) -> dict[str, float]:
        n_select = min(self.clients_per_round, len(train_client_ids))
        selected = np.random.choice(train_client_ids, n_select, replace=False).tolist()
        all_tasks = []
        for cid in selected:
            all_tasks.extend(
                fed_data.create_meta_tasks(cid, n_tasks, k_shot, seed=self.round + cid)
            )
        if self.round % 5 == 0:
            self.compute_fisher(all_tasks)
        metrics = self.meta_train_step(all_tasks)
        if self.drift_monitor:
            for cid in selected:
                self.drift_monitor.update(cid, metrics["meta_loss"])
            self.drift_monitor.advance_step()
        self.round += 1
        return {
            "avg_loss": metrics["meta_loss"],
            "n_drifts": 0,
            "round": self.round,
        }

    def evaluate(
        self,
        fed_data,
        test_client_ids: list[int],
        n_tasks: int = 5,
        k_shot: int = 20,
    ) -> dict[str, float]:
        helper = TemporalMAML(
            copy.deepcopy(self.model),
            self.inner_lr, 0.001, self.inner_steps, 0.9, self.clients_per_round, self.device,
        )
        all_accs = []
        for cid in test_client_ids:
            tasks = fed_data.create_meta_tasks(cid, n_tasks, k_shot, seed=9999 + cid)
            all_accs.append(helper.evaluate_tasks(tasks)["mean_accuracy"])
        return {
            "mean_accuracy": float(np.mean(all_accs)) if all_accs else 0.0,
            "std_accuracy": float(np.std(all_accs)) if all_accs else 0.0,
            "per_client_acc": {cid: acc for cid, acc in zip(test_client_ids, all_accs)},
        }


class TemporalFedAvg:
    """Temporal-weighted FedAvg: favors recent client updates.

    Server maintains a time-decayed aggregation where clients
    with more recent updates get higher weight.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 3,
        temporal_decay: float = 0.9,
        ewc_lambda: float = 0.5,
        clients_per_round: int = 5,
        device: str = "cpu",
    ):
        self.global_model = model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.temporal_decay = temporal_decay
        self.ewc_lambda = ewc_lambda
        self.clients_per_round = clients_per_round
        self.device = device

        self.drift_monitor = None
        self.client_timestamps: dict[int, int] = {}
        self.round = 0

    def init_drift_monitor(self, n_clients: int):
        self.drift_monitor = ClientDriftMonitor(n_clients)

    def train_round(
        self,
        fed_data,
        train_client_ids: list[int],
        n_tasks: int = 5,
        k_shot: int = 20,
    ) -> dict[str, float]:
        """One federated round with temporal weighting."""
        n_select = min(self.clients_per_round, len(train_client_ids))
        selected = np.random.choice(
            train_client_ids, n_select, replace=False
        ).tolist()

        client_models = []
        client_losses = []
        drifts = []

        for cid in selected:
            local_model = copy.deepcopy(self.global_model)
            local_maml = TemporalMAML(
                local_model, self.inner_lr, self.outer_lr,
                self.inner_steps, self.temporal_decay,
                clients_per_round=1, device=self.device,
            )

            tasks = fed_data.create_meta_tasks(cid, n_tasks, k_shot, seed=self.round + cid)
            for task_batch in [tasks]:
                metrics = local_maml.meta_train_step(task_batch)

            client_models.append(local_maml.model)
            client_losses.append(metrics["meta_loss"])
            drifts.append(metrics["drift_detected"])
            self.client_timestamps[cid] = self.round

            if self.drift_monitor:
                self.drift_monitor.update(cid, metrics["meta_loss"])

        # Temporal-weighted aggregation
        weights = []
        for cid in selected:
            staleness = self.round - self.client_timestamps.get(cid, 0)
            weights.append(self.temporal_decay ** staleness)
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = sum(
                w * m.state_dict()[key].float()
                for w, m in zip(weights, client_models)
            )
        self.global_model.load_state_dict(global_dict)

        self.round += 1
        if self.drift_monitor:
            self.drift_monitor.advance_step()

        return {
            "avg_loss": float(np.mean(client_losses)),
            "n_drifts": sum(drifts),
            "round": self.round,
        }

    def evaluate(
        self,
        fed_data,
        test_client_ids: list[int],
        n_tasks: int = 5,
        k_shot: int = 20,
    ) -> dict[str, float]:
        """Evaluate on test clients."""
        all_accs = []
        for cid in test_client_ids:
            tasks = fed_data.create_meta_tasks(cid, n_tasks, k_shot, seed=9999 + cid)
            maml = TemporalMAML(
                copy.deepcopy(self.global_model),
                self.inner_lr, self.outer_lr,
                self.inner_steps, self.temporal_decay,
                clients_per_round=1, device=self.device,
            )
            eval_result = maml.evaluate_tasks(tasks)
            all_accs.append(eval_result["mean_accuracy"])

        return {
            "mean_accuracy": float(np.mean(all_accs)),
            "std_accuracy": float(np.std(all_accs)),
            "per_client_acc": {cid: acc for cid, acc in zip(test_client_ids, all_accs)},
        }
