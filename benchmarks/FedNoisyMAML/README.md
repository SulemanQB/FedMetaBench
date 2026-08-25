# FedNoisyMAML: Federated Meta-Learning Under Heterogeneous Label Noise

Part of **[FedMetaBench](../../README.md)** — GitHub-ready federated meta-learning demos.

## Overview

This project studies whether **MAML-based personalization provides robustness** to heterogeneous label noise in federated learning. Each client has a different label noise rate (0% to 50%), and we compare FedMAML against FedAvg.

**Key finding** (synthetic Gaussian mixture, 10 seeds): FedMAML consistently outperforms FedAvg (**p=0.002**) across noise levels.

## Method

- **FedMAML**: Inner-loop adaptation on noisy support data, meta-gradient aggregation across clients
- **FedAvg** (baseline): Standard federated averaging with local SGD on noisy labels
- **Data**: Synthetic 2D Gaussian mixture (default); optional MNIST/CIFAR via `dataset` config
- **Noise model**: Heterogeneous — client i gets noise rate = i/(n−1) × noise_max
- **Evaluation**: Per-client accuracy on clean held-out test data after adaptation

## Quick Start

```bash
pip install -r requirements.txt

# Full multi-seed experiment
python run_experiment.py --config configs/default.yaml

# Fast smoke (CPU)
python run_experiment.py --config configs/smoke.yaml

python tests/smoke_test.py
```

## Project Structure

```
├── run_experiment.py
├── configs/{default,smoke}.yaml
├── src/{algorithms,datasets,models,image_data}.py
├── tests/smoke_test.py
└── requirements.txt
```

## Limitations

- Default evidence is synthetic; treat image results as optional extensions.
- For journal-depth robustness claims, add stronger noise-robust FL baselines.
