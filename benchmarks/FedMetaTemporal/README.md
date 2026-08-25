# FedMetaTemporal

Federated learning under **temporal concept drift** (sudden / gradual / recurring / incremental).

## Algorithms

| Config `algorithm.name` | Description |
|-------------------------|-------------|
| `temporal_fedavg` | Time-decayed aggregation of recent client updates |
| `temporal_maml` | Temporally weighted MAML outer loop + drift detector |
| `ewc_maml` | MAML + Elastic Weight Consolidation |

> **Fix (2026):** `run_experiment.py` now instantiates the algorithm named in the YAML. Older sweeps that assumed all names ran TemporalFedAvg should be re-run.

## Quick start

```bash
pip install -r requirements.txt
python run_experiment.py --config configs/default.yaml
python run_experiment.py --config configs/smoke.yaml   # uses temporal_maml
```

## Data

Synthetic drifting client streams (`src/datasets/temporal_streams.py`). Optional Neural CDE encoder via `model.type: neural_cde`.

## Notes

- Drift monitoring via Page–Hinkley / client drift monitor.
- Results: `results/<name>_results.json` with `final.mean_accuracy` and `drift_summary`.
- Part of [FedMetaBench](../../README.md). Journal-track drift work: **MetaDrift**.
