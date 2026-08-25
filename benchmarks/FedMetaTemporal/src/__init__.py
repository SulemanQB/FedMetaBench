"""FedMeta-Temporal+: Federated Meta-Learning for Temporal Concept Drift.

Paper 4a: Neural CDE + MAML for irregular time series
Paper 4b: EWC + modular networks + FLTA comparison

Key innovations:
- Neural CDE encoder for irregular time series in federated meta-learning
- Page-Hinkley drift detection for triggering re-adaptation
- Temporal-weighted FedAvg favoring recent client updates
- EWC regularization to prevent catastrophic forgetting during drift
"""
