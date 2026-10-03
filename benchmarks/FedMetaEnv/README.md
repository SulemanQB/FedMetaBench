# FedMetaEnv

FedMetaEnv evaluates federated personalization for environmental sensing. Each monitoring station is a client; leave-one-station-out (LOSO) evaluation measures how quickly a model adapts to a held-out station.

## Workflow and Methods

The loader uses Beijing PRSA CSVs when found and otherwise generates twelve synthetic station streams with spatial heterogeneity. The runner trains on all but one station, evaluates before and after K-day adaptation, and reports regression metrics.

| Config name | Role |
| --- | --- |
| `fed_env_maml` | Federated MAML for cold-start stations |
| `fedavg` | Standard federated averaging |
| `local_only` | Target-station-only baseline |

## Run

```bash
python -m pip install -r ../../requirements.txt
python run_experiment.py --config configs/smoke.yaml
python run_experiment.py --config configs/default.yaml
python tests/smoke_test.py
```

The smoke config evaluates the first two stations and uses `k_days_list: [3]`. Results are written to `results/<experiment>_results.json`, with keys such as `k3` matching the configured evaluation budget.

## Data and Configuration

Place files named `PRSA_Data_<Station>_20130301-20170228.csv` under `data/`, `data/beijing_air_quality/`, or `data/PRSA_Data_20130301-20170228/`. For an external directory, set `FEDMETAENV_DATA_DIR`. Missing CSVs intentionally trigger the synthetic fallback. YAML controls window length, pollutant target, station budget, model, rounds, and K-day evaluation list.

## Technical Highlights and Limits

- LOSO makes cold-start generalization explicit rather than mixing train and test stations.
- Metrics include MSE, MAE, RMSE, and adaptation improvement.
- Synthetic streams are not a substitute for real sensor validation; the sibling FedSense research track is intentionally outside this repository.

See the [root README](../../README.md) for the full suite and project scope.
