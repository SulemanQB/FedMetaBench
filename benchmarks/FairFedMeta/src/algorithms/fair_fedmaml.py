"""FairFedMAML — Federated FairMAML with fairness-aware aggregation.

Combines:
1. FairMAML inner/outer loop per client
2. Federated aggregation of model params + fairness weights λ
3. DRO-inspired server-side lambda update using cross-client group info
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fair_maml import FairMAML


class FairFedMAML:
    """Federated FairMAML: server orchestrates fairness-aware meta-learning.

    Server maintains:
    - Global model θ
    - Global fairness weights λ_g (meta-learned)
    - Per-hospital fairness statistics

    Each round:
    1. Server sends θ, λ to selected hospitals
    2. Each hospital runs FairMAML inner/outer on local data
    3. Server aggregates model updates (weighted by hospital size)
    4. Server aggregates λ updates (DRO over cross-hospital group statistics)
    """

    def __init__(
        self,
        model: nn.Module,
        n_groups: int,
        num_rounds: int = 100,
        clients_per_round: int = 10,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        lambda_lr: float = 0.01,
        inner_steps: int = 1,
        local_meta_steps: int = 5,
        fairness_penalty: float = 1.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.global_model = model.to(device)
        self.n_groups = n_groups
        self.clients_per_round = clients_per_round
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.lambda_lr = lambda_lr
        self.inner_steps = inner_steps
        self.local_meta_steps = local_meta_steps
        self.fairness_penalty = fairness_penalty
        self.device = device

        # Global fairness weights
        self.global_log_lambdas = torch.zeros(n_groups, device=device)

        self.current_round = 0

    def train_round(
        self,
        fed_dataset,
        selected_hospitals: list[int],
    ) -> dict[str, float]:
        """Execute one federated round."""
        client_results = []
        client_weights = []

        for hosp_id in selected_hospitals:
            result = self._client_update(fed_dataset, hosp_id)
            client_results.append(result)
            client_weights.append(result["num_samples"])

        # Normalize weights
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]

        # Aggregate model parameters
        self._aggregate_models(client_results, client_weights)

        # Aggregate fairness weights
        self._aggregate_lambdas(client_results, client_weights)

        self.current_round += 1

        metrics = {
            "round": self.current_round,
            "avg_loss": sum(r["meta_loss"] * w for r, w in zip(client_results, client_weights)),
            "avg_fairness_penalty": sum(
                r["fairness_penalty"] * w for r, w in zip(client_results, client_weights)
            ),
            "global_lambdas": F.softmax(self.global_log_lambdas, dim=0).cpu().numpy().tolist(),
        }

        # Aggregate per-group accuracies
        for g in range(self.n_groups):
            key = f"group_{g}_acc"
            vals = [r[key] for r in client_results if key in r]
            if vals:
                metrics[f"group_{g}_acc"] = float(np.mean(vals))

        return metrics

    def _client_update(self, fed_dataset, hospital_id: int) -> dict:
        """Run FairMAML on a single hospital's data."""
        client_model = copy.deepcopy(self.global_model)

        fair_maml = FairMAML(
            model=client_model,
            n_groups=self.n_groups,
            inner_lr=self.inner_lr,
            outer_lr=self.outer_lr,
            lambda_lr=self.lambda_lr,
            inner_steps=self.inner_steps,
            fairness_penalty=self.fairness_penalty,
            device=self.device,
        )
        # Initialize with global lambdas
        fair_maml.log_lambdas.data = self.global_log_lambdas.clone()

        client_data = fed_dataset.get_client_data(hospital_id)
        n = len(client_data["labels"])

        # Create mini-tasks from client data (split into support/query)
        accumulated_metrics = []
        for _ in range(self.local_meta_steps):
            task = self._create_task(client_data, hospital_id, fed_dataset)
            metrics = fair_maml.outer_step([task])
            accumulated_metrics.append(metrics)

        # Return updated model + lambdas
        final = accumulated_metrics[-1] if accumulated_metrics else {}
        final["model_state"] = {k: v.cpu() for k, v in fair_maml.model.state_dict().items()}
        final["log_lambdas"] = fair_maml.log_lambdas.detach().cpu()
        final["num_samples"] = n
        return final

    def _create_task(self, client_data: dict, hospital_id: int, fed_dataset) -> dict:
        """Create a support/query task from client data."""
        support, query = fed_dataset.get_group_stratified_split(
            hospital_id, k_shot=5, seed=self.current_round + hash(hospital_id) % 1000,
        )
        return {
            "support_features": torch.FloatTensor(support["features"]),
            "support_labels": torch.LongTensor(support["labels"]),
            "support_groups": torch.LongTensor(support["group_ids"]),
            "query_features": torch.FloatTensor(query["features"]),
            "query_labels": torch.LongTensor(query["labels"]),
            "query_groups": torch.LongTensor(query["group_ids"]),
        }

    def _aggregate_models(self, client_results: list[dict], weights: list[float]):
        """FedAvg-style aggregation of model parameters."""
        avg_state = {}
        for key in client_results[0]["model_state"]:
            avg_state[key] = sum(
                w * r["model_state"][key].float()
                for w, r in zip(weights, client_results)
            )
        self.global_model.load_state_dict(avg_state)

    def _aggregate_lambdas(self, client_results: list[dict], weights: list[float]):
        """Aggregate fairness weights across clients."""
        avg_lambdas = sum(
            w * r["log_lambdas"] for w, r in zip(weights, client_results)
        )
        self.global_log_lambdas = avg_lambdas.to(self.device)
