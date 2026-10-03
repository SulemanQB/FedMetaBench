# FedNoisyMAML

FedNoisyMAML tests whether client-specific adaptation is useful when federated clients have different label-noise rates. It compares FedMAML with FedAvg on a synthetic classification task and optionally on MNIST or CIFAR-10.

## Workflow

The runner creates clean synthetic data, partitions it across clients, injects client-specific training-label noise, trains both methods, and evaluates each client on clean held-out data. The noise rate increases linearly from zero to `noise_max` across clients.

| Item | Default |
| --- | --- |
| Model | MLP for Gaussian data; CNN for images |
| Algorithms | FedMAML and FedAvg |
| Metric | Per-client and overall accuracy |
| Reproducibility | YAML seeds, NumPy and PyTorch seeding |

## Run

```bash
python -m pip install -r ../../requirements.txt
python run_experiment.py --config configs/smoke.yaml
python tests/smoke_test.py
```

Use `configs/default.yaml` for the longer ten-seed Gaussian experiment. Select `dataset: mnist` or `dataset: cifar10` only after installing the optional image dependencies in `requirements.txt` and setting `DATA_ROOT` if a custom dataset directory is needed.

## Configuration and Output

`n_clients`, `noise_max`, `inner_steps`, learning rates, rounds, seeds, dataset, and device are configured in YAML. Results are written as JSON under `results/`, which is ignored by Git. Smoke mode uses four clients, five rounds, one seed, synthetic data, and CPU.

## Technical Highlights

- Client-specific label corruption makes robustness measurable rather than implicit.
- FedMAML and FedAvg share the same generated federation and clean evaluation contract.
- The statistical report includes per-noise accuracy, deltas, sign-test p-values, and degradation slopes.

## Limitations

The default task is synthetic and does not establish image-dataset or deployment performance. Stronger noise-robust baselines and repeated real-dataset experiments are needed before making research claims. See the [root README](../../README.md) for the full suite and reproducibility guidance.
