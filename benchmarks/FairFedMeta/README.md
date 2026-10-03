# FairFedMeta

FairFedMeta studies group fairness in federated clinical-style classification. Hospitals are clients, demographic groups are represented explicitly, and configurable baselines are compared on synthetic eICU-like sequences by default.

## Workflow and Methods

The runner loads or generates hospital sequences, creates meta-train and held-out hospital splits, builds support/query tasks, trains the selected algorithm, and records overall accuracy, worst-group accuracy, and the accuracy gap.

| Config name | Role |
| --- | --- |
| `fair_fedmaml` | Fairness-constrained federated meta-learning |
| `fair_fedavg` | Group-aware FedAvg baseline |
| `agnostic_fair` | Worst-group client reweighting baseline |
| `fedavg`, `per_fedavg` | Standard and personalization baselines |

## Run

```bash
python -m pip install -r ../../requirements.txt
python run_experiment.py --config configs/smoke.yaml
python run_experiment.py --config configs/default.yaml algorithm.name=fair_fedmaml
python tests/smoke_test.py
```

The smoke config uses synthetic data, CPU, three rounds, and a small hospital federation. Results are written to `results/<experiment>_results.json`.

## Data and Configuration

Synthetic data contains no PHI. A processed eICU HDF5 file can be supplied through `dataset.data_path`; the optional `h5py` dependency and expected layout are defined in `src/datasets/eicu.py`. YAML controls hospital count, sequence length, group count, model, fairness penalties, client sampling, rounds, and seed.

## Technical Highlights and Limits

- Explicit group IDs flow through support/query task construction and fairness metrics.
- LSTM and GRU clinical sequence models share the same runner contract.
- The demo is not a clinical validation study: synthetic data is the default, protected-group semantics are simplified, and broader calibration and fairness analysis remain future work.

See the [root README](../../README.md) for the suite architecture and project scope.
