# FedMetaBench

A curated **GitHub-ready** benchmark suite for federated meta-learning prototypes.

> **Scope.** This repo packages four self-contained benchmarks for portfolio / applied demos.  
> Journal-track projects (**FedSense**, **MetaDrift**, **MetaForgetting**, …) stay outside this suite.

| Benchmark | Theme | Status |
|-----------|--------|--------|
| [FedNoisyMAML](benchmarks/FedNoisyMAML) | Heterogeneous label noise vs FedAvg | Demo-ready |
| [FairFedMeta](benchmarks/FairFedMeta) | Fairness-aware FedMAML (synthetic clinical) | Runnable |
| [FedMetaEnv](benchmarks/FedMetaEnv) | Environmental LOSO (Beijing air quality) | Runnable |
| [FedMetaTemporal](benchmarks/FedMetaTemporal) | Temporal drift + TemporalFedAvg / EWC-MAML / Temporal-MAML | Runnable |

## Requirements

- Python **≥ 3.10** (tested with 3.11/3.12)
- PyTorch ≥ 2.0, NumPy, PyYAML, SciPy, pandas (for real Beijing CSVs)

```bash
cd FedMetaBench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
# Smoke all four (CPU, few rounds)
bash scripts/run_smoke_all.sh

# Or run one benchmark
cd benchmarks/FedNoisyMAML && python run_experiment.py --config configs/default.yaml
cd benchmarks/FairFedMeta && python run_experiment.py --config configs/default.yaml
cd benchmarks/FedMetaEnv && python run_experiment.py --config configs/default.yaml
cd benchmarks/FedMetaTemporal && python run_experiment.py --config configs/default.yaml
```

## Design notes

- Each benchmark is **independent** (`run_experiment.py` + `src/` + `configs/`).
- Results write to `benchmarks/<name>/results/*.json`.
- Synthetic data is the default fallback when real datasets are missing; FedMetaEnv will use Beijing PRSA CSVs if present under `data/` or linked from FedSense.
- Algorithm names in YAML are honored (FedMetaTemporal no longer hard-codes TemporalFedAvg).

## What is *not* in this repo

| Project | Reason |
|---------|--------|
| FedSense | Journal-track air-quality FL |
| MetaDrift / MetaForgetting | Journal-track novelty candidates |
| EvalFedMeta / FedMAMLBudget | Separate research tracks |

## License

MIT — see [LICENSE](LICENSE).
