# FedMetaBench scope

## In this repo (GitHub / industry portfolio)

Four applied federated meta-learning demos:

1. **FedNoisyMAML** — label noise robustness
2. **FairFedMeta** — fairness + FedMAML
3. **FedMetaEnv** — environmental LOSO
4. **FedMetaTemporal** — temporal drift algorithms

## Outside this repo (journal track)

Do **not** fold these into FedMetaBench without a separate decision:

- **FedSense** — air-quality FL system + full manuscript
- **MetaDrift** — adaptation-gap drift detection (highest novelty)
- **MetaForgetting** — long-horizon meta-forgetting metrics
- **EvalFedMeta** / **FedMAMLBudget** — broader evaluation / budget studies

## Fixes applied (Aug 2026)

- FedMetaTemporal: algorithm factory respects `algorithm.name`
- FairFedMeta: support/query `group_ids` alignment; Reptile-style Per-FedAvg; input_dim sync
- FedMetaEnv: multi-path Beijing CSV discovery
- Smoke configs + `scripts/run_smoke_all.sh`
- READMEs for all four packages
