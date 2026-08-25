# FairFedMeta

Fairness-aware federated meta-learning for clinical-style hospital splits.

## Idea

Combine **FedMAML-style personalization** with **group fairness** (demographic groups across hospitals). Compares:

- `fair_fedmaml` — fairness-constrained federated MAML
- `fair_fedavg` — FedAvg with equal group weights
- `agnostic_fair` — worst-group client reweighting (Mohri-style)
- `fedavg`, `per_fedavg` — baselines

## Quick start

```bash
pip install -r requirements.txt
python run_experiment.py --config configs/default.yaml
python run_experiment.py --config configs/smoke.yaml   # short CPU smoke
# Override algorithm:
python run_experiment.py --config configs/default.yaml algorithm.name=fair_fedmaml
```

## Data

- Default: **synthetic** eICU-like placeholder (no PHI).
- Optional: place processed eICU at `data/eicu_processed.h5` (see `src/datasets/eicu.py`).

## Notes

- Metrics: overall accuracy, worst-group accuracy, accuracy gap.
- Results: `results/<experiment>_results.json`
- Part of [FedMetaBench](../../README.md) for GitHub demos; not the journal track.
