"""FairMAML — MAML with meta-learned per-group fairness weights.

Core contribution of Paper 2 (FairFedMeta):
    Meta-objective: min_θ Σ_i [ Σ_g λ_g · L_g(f_{θ_i'}, D_i^query) ]
    where λ_g are meta-learned per-group weights that ensure equitable
    performance across demographic groups.

    θ_i' = θ - α · ∇_θ Σ_g L_g(f_θ, D_i^support)   (inner loop)
    λ_g updated via meta-gradient to minimize worst-group performance gap.
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


class FairMAML:
    """MAML with meta-learned per-group fairness weights λ_g.

    Key innovation: λ_g are treated as meta-parameters optimized in the
    outer loop to minimize the fairness gap (max-group - min-group performance).
    """

    def __init__(
        self,
        model: nn.Module,
        n_groups: int,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        lambda_lr: float = 0.01,
        inner_steps: int = 1,
        fairness_penalty: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.n_groups = n_groups
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.fairness_penalty = fairness_penalty
        self.device = device

        # Meta-learned fairness weights (initialized uniform)
        self.log_lambdas = nn.Parameter(
            torch.zeros(n_groups, device=device)
        )

        # Outer optimizer for model params + lambda
        self.meta_optimizer = torch.optim.Adam(
            list(self.model.parameters()) + [self.log_lambdas],
            lr=outer_lr,
        )
        self.lambda_optimizer = torch.optim.SGD(
            [self.log_lambdas], lr=lambda_lr,
        )

    @property
    def lambdas(self) -> torch.Tensor:
        """Normalized fairness weights via softmax."""
        return F.softmax(self.log_lambdas, dim=0) * self.n_groups

    def inner_loop(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        support_groups: torch.Tensor,
    ) -> nn.Module:
        """MAML inner loop: adapt model on support set."""
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()

        for _ in range(self.inner_steps):
            out = adapted_model(support_features)
            # Group-weighted loss for inner loop
            loss = self._group_weighted_loss(out, support_labels, support_groups)
            grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=True)
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data = param.data - self.inner_lr * grad

        return adapted_model

    def outer_step(
        self,
        client_tasks: list[dict],
    ) -> dict[str, float]:
        """One meta-update step across multiple client tasks.

        Args:
            client_tasks: List of dicts, each with:
                support_features, support_labels, support_groups,
                query_features, query_labels, query_groups
        """
        self.meta_optimizer.zero_grad()

        total_loss = 0.0
        total_fair_loss = 0.0
        group_accs = {g: [] for g in range(self.n_groups)}

        for task in client_tasks:
            sf = task["support_features"].to(self.device)
            sl = task["support_labels"].to(self.device)
            sg = task["support_groups"].to(self.device)
            qf = task["query_features"].to(self.device)
            ql = task["query_labels"].to(self.device)
            qg = task["query_groups"].to(self.device)

            # Inner loop adaptation
            adapted = self.inner_loop(sf, sl, sg)

            # Outer loss on query set with fairness weighting
            query_out = adapted(qf)
            query_loss = self._group_weighted_loss(query_out, ql, qg)

            # Fairness penalty: minimize max-min group performance gap
            fair_penalty = self._fairness_penalty(query_out, ql, qg)

            task_loss = query_loss + self.fairness_penalty * fair_penalty
            total_loss += task_loss
            total_fair_loss += fair_penalty.item()

            # Track per-group accuracy
            preds = query_out.argmax(dim=-1) if query_out.dim() > 1 else (query_out.squeeze() > 0).long()
            for g in range(self.n_groups):
                g_mask = qg == g
                if g_mask.sum() > 0:
                    g_acc = (preds[g_mask] == ql[g_mask]).float().mean().item()
                    group_accs[g].append(g_acc)

        # Backward + update
        avg_loss = total_loss / len(client_tasks)
        avg_loss.backward()
        self.meta_optimizer.step()

        # Additional lambda update via DRO-inspired ascent
        self._update_lambdas_dro(client_tasks)

        metrics = {
            "meta_loss": avg_loss.item(),
            "fairness_penalty": total_fair_loss / len(client_tasks),
            "lambdas": self.lambdas.detach().cpu().numpy().tolist(),
        }
        for g in range(self.n_groups):
            if group_accs[g]:
                metrics[f"group_{g}_acc"] = float(np.mean(group_accs[g]))

        return metrics

    def _group_weighted_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor,
    ) -> torch.Tensor:
        """Compute λ_g-weighted loss across demographic groups."""
        lambdas = self.lambdas
        total_loss = torch.tensor(0.0, device=self.device)

        if logits.dim() > 1 and logits.shape[-1] > 1:
            criterion = nn.CrossEntropyLoss(reduction="none")
        else:
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            logits = logits.squeeze()
            labels = labels.float()

        per_sample_loss = criterion(logits, labels)

        for g in range(self.n_groups):
            g_mask = groups == g
            if g_mask.sum() > 0:
                g_loss = per_sample_loss[g_mask].mean()
                total_loss += lambdas[g] * g_loss

        return total_loss / self.n_groups

    def _fairness_penalty(
        self, logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor,
    ) -> torch.Tensor:
        """Max-min fairness penalty: penalize gap between best and worst group."""
        group_losses = []
        if logits.dim() > 1 and logits.shape[-1] > 1:
            criterion = nn.CrossEntropyLoss(reduction="none")
        else:
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            logits = logits.squeeze()
            labels = labels.float()

        per_sample_loss = criterion(logits, labels)

        for g in range(self.n_groups):
            g_mask = groups == g
            if g_mask.sum() > 0:
                group_losses.append(per_sample_loss[g_mask].mean())

        if len(group_losses) < 2:
            return torch.tensor(0.0, device=self.device)

        stacked = torch.stack(group_losses)
        return stacked.max() - stacked.min()

    def _update_lambdas_dro(self, client_tasks: list[dict]):
        """DRO-inspired lambda update: increase weight for worst-performing groups."""
        group_losses = torch.zeros(self.n_groups, device=self.device)
        group_counts = torch.zeros(self.n_groups, device=self.device)

        for task in client_tasks:
            sf = task["support_features"].to(self.device)
            sl = task["support_labels"].to(self.device)
            qf = task["query_features"].to(self.device)
            ql = task["query_labels"].to(self.device)
            qg = task["query_groups"].to(self.device)

            # Simple gradient-based adaptation for DRO loss estimation
            adapted = copy.deepcopy(self.model)
            adapted.train()
            opt = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)
            for _ in range(self.inner_steps):
                opt.zero_grad()
                out = adapted(sf)
                loss = F.cross_entropy(out, sl)
                loss.backward()
                opt.step()

            adapted.eval()
            with torch.no_grad():
                out = adapted(qf)

            if out.dim() > 1 and out.shape[-1] > 1:
                criterion = nn.CrossEntropyLoss(reduction="none")
            else:
                criterion = nn.BCEWithLogitsLoss(reduction="none")
                out = out.squeeze()
                ql = ql.float()

            losses = criterion(out, ql)
            for g in range(self.n_groups):
                g_mask = qg == g
                if g_mask.sum() > 0:
                    group_losses[g] += losses[g_mask].sum()
                    group_counts[g] += g_mask.sum()

        # Normalize
        for g in range(self.n_groups):
            if group_counts[g] > 0:
                group_losses[g] /= group_counts[g]

        # DRO update: λ_g ← λ_g * exp(η * loss_g) then normalize
        eta = 0.1
        self.log_lambdas.data += eta * group_losses

    def evaluate_fairness(
        self,
        test_tasks: list[dict],
        inner_steps: int | None = None,
    ) -> dict[str, float]:
        """Comprehensive fairness evaluation on test tasks."""
        inner_steps = inner_steps or self.inner_steps
        group_accs = {g: [] for g in range(self.n_groups)}
        group_losses = {g: [] for g in range(self.n_groups)}
        overall_accs = []

        for task in test_tasks:
            sf = task["support_features"].to(self.device)
            sl = task["support_labels"].to(self.device)
            sg = task["support_groups"].to(self.device)
            qf = task["query_features"].to(self.device)
            ql = task["query_labels"].to(self.device)
            qg = task["query_groups"].to(self.device)

            # Adapt
            adapted = self.inner_loop(sf, sl, sg)
            adapted.eval()

            with torch.no_grad():
                out = adapted(qf)
                if out.dim() > 1 and out.shape[-1] > 1:
                    preds = out.argmax(dim=-1)
                    criterion = nn.CrossEntropyLoss(reduction="none")
                else:
                    preds = (out.squeeze() > 0).long()
                    criterion = nn.BCEWithLogitsLoss(reduction="none")
                    out = out.squeeze()
                    ql_loss = ql.float()
                    losses = criterion(out, ql_loss)

                overall_accs.append((preds == ql).float().mean().item())

                for g in range(self.n_groups):
                    g_mask = qg == g
                    if g_mask.sum() > 0:
                        g_acc = (preds[g_mask] == ql[g_mask]).float().mean().item()
                        group_accs[g].append(g_acc)

        results = {
            "overall_accuracy": float(np.mean(overall_accs)),
            "lambdas": self.lambdas.detach().cpu().numpy().tolist(),
        }

        # Per-group metrics
        valid_group_accs = []
        for g in range(self.n_groups):
            if group_accs[g]:
                g_mean = float(np.mean(group_accs[g]))
                results[f"group_{g}_accuracy"] = g_mean
                valid_group_accs.append(g_mean)

        if len(valid_group_accs) >= 2:
            results["accuracy_gap"] = max(valid_group_accs) - min(valid_group_accs)
            results["worst_group_accuracy"] = min(valid_group_accs)
            results["best_group_accuracy"] = max(valid_group_accs)
            results["equalized_accuracy_std"] = float(np.std(valid_group_accs))

        return results
