# FedMetaBench

Federated learning trains a shared model across clients that cannot pool their raw data. Those clients still disagree with each other: label noise varies by site, demographic groups are uneven, a new sensor has almost no history, and the distribution moves over time. FedMetaBench is a PyTorch suite of four independent benchmarks that compare meta-learning-style personalization (MAML and related variants) with FedAvg-style baselines under those shifts. A clean clone runs on synthetic data. Beijing multi-site air-quality CSVs and a processed eICU file are optional inputs. Each run writes JSON under that benchmark's `results/` directory. This page does not tabulate scores.

## What's included

| Benchmark | Question | Methods | Data on a clean clone | Smoke (`configs/smoke.yaml`) |
|-----------|----------|---------|------------------------|------------------------------|
| [FedNoisyMAML](benchmarks/FedNoisyMAML) | Does per-client adaptation hold up when label noise differs across clients? | FedMAML and FedAvg (both run on every experiment) | Synthetic Gaussian mixture. MNIST and CIFAR are optional and need `torchvision`. | 4 clients, 5 rounds, 1 seed, CPU |
| [FairFedMeta](benchmarks/FairFedMeta) | Can federated meta-learning narrow worst-group gaps across hospital-style splits? | `fair_fedmaml`, `fair_fedavg`, `agnostic_fair`, `fedavg`, `per_fedavg` | Synthetic clinical sequences (no PHI). Optional file: `data/eicu_processed.h5`. | `fair_fedavg`, 3 rounds, CPU |
| [FedMetaEnv](benchmarks/FedMetaEnv) | How quickly can a model cold-start on a held-out monitoring station (leave-one-station-out)? | `fed_env_maml`, `fedavg`, `local_only` | Synthetic 12-station streams. UCI Beijing PRSA CSVs are used when a searched path contains them; see that package README. | `fedavg`, 2 rounds, first 2 stations, CPU |
| [FedMetaTemporal](benchmarks/FedMetaTemporal) | How do federated methods behave under temporal concept drift? | `temporal_fedavg`, `temporal_maml`, `ewc_maml` | Synthetic drifting client streams (`sudden`, `gradual`, `recurring`, or `incremental`). | `temporal_maml`, sudden drift, 3 rounds, CPU |

Per-benchmark READMEs describe the methods. Smoke and default configs are the reproducible entry points in this repository.

## How to run

Python 3.10 or newer. The root [`requirements.txt`](requirements.txt) is what the smoke script needs: PyTorch ≥ 2.0, NumPy, PyYAML, SciPy, pandas, tqdm, and scikit-learn.

```bash
git clone https://github.com/SulemanQB/FedMetaBench.git
cd FedMetaBench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
bash scripts/run_smoke_all.sh
```

A finished smoke run prints:

```text
[FedMetaBench] All smoke runs completed.
```

Checked on Python 3.12.3: a new virtualenv, `pip install -r requirements.txt`, and `bash scripts/run_smoke_all.sh` finish with that line. The same environment passes each package's `python tests/smoke_test.py`. On Linux the default PyTorch wheel from the root requirements file includes CUDA libraries; the smoke configs still set `device: cpu` and complete without a GPU.

`scripts/run_smoke_all.sh` calls `python3` (override with `PYTHON=...`) and runs `run_experiment.py --config configs/smoke.yaml` in each package. JSON lands in `benchmarks/<name>/results/` and is gitignored.

One benchmark at a time:

```bash
cd benchmarks/FedNoisyMAML && python run_experiment.py --config configs/smoke.yaml
cd benchmarks/FairFedMeta && python run_experiment.py --config configs/smoke.yaml
cd benchmarks/FedMetaEnv && python run_experiment.py --config configs/smoke.yaml
cd benchmarks/FedMetaTemporal && python run_experiment.py --config configs/smoke.yaml
```

`configs/default.yaml` is the longer demo. FedNoisyMAML's default config requests CUDA, uses 10 seeds and 150 rounds on the Gaussian mixture, and switches to CPU when CUDA is unavailable. Image datasets (`mnist`, `cifar10`) need the extra packages in [`benchmarks/FedNoisyMAML/requirements.txt`](benchmarks/FedNoisyMAML/requirements.txt), including `torchvision`. A real eICU file for FairFedMeta needs `h5py` from that package's requirements file. `matplotlib` and `seaborn` are listed per benchmark and are unused by the smoke script.

Each package also has a shorter unit check: `python tests/smoke_test.py` from that package directory. The end-to-end check is `bash scripts/run_smoke_all.sh`.

## Layout

Each benchmark stands alone (`run_experiment.py`, `src/`, `configs/`, `tests/`). FedMetaTemporal builds the algorithm named in the YAML (`temporal_fedavg`, `temporal_maml`, or `ewc_maml`).

## What is *not* in this repo

| Project | Reason |
|---------|--------|
| FedSense | Journal-track air-quality FL |
| MetaDrift / MetaForgetting | Journal-track novelty candidates |
| EvalFedMeta / FedMAMLBudget | Separate research tracks |

## License

MIT — see [LICENSE](LICENSE).
