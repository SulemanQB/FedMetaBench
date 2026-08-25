# FedMetaEnv

Federated meta-learning for **environmental sensing** with leave-one-station-out (LOSO) evaluation on Beijing multi-site air quality.

## Idea

Each monitoring station is a client. Train on N−1 stations, adapt with K days of data on the held-out station, and compare:

- `fed_env_maml` — federated MAML for cold-start stations
- `fedavg` — standard FedAvg
- `local_only` — train only on the target station

## Quick start

```bash
pip install -r requirements.txt
# Optional: link real UCI Beijing CSVs
# mkdir -p data && ln -s /path/to/PRSA_Data_20130301-20170228 data/
python run_experiment.py --config configs/default.yaml
python run_experiment.py --config configs/smoke.yaml
```

## Data

Looks for `PRSA_Data_<Station>_*.csv` under:

1. `./data/beijing_air_quality/`
2. `./data/PRSA_Data_20130301-20170228/`
3. `./data/`
4. FedSense data path (if present on this machine)

Falls back to synthetic station streams if CSVs are missing.

## Notes

- LOSO metrics: pre/post-adapt MSE, MAE, RMSE, improvement.
- Sibling journal project: **FedSense** (separate repo / paper track).
- Part of [FedMetaBench](../../README.md).
