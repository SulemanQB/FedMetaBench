#!/usr/bin/env bash
# CPU smoke tests for all FedMetaBench packages (short configs).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
export PYTHONPATH=""

echo "[FedMetaBench] Using: $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"

run_one() {
  local name="$1"
  local cfg="$2"
  echo ""
  echo "=== Smoke: $name ==="
  cd "$ROOT/benchmarks/$name"
  $PY run_experiment.py --config "$cfg"
  echo "OK: $name"
}

# Prefer smoke YAML if present, else default with override via short configs
run_one FedNoisyMAML configs/smoke.yaml
run_one FairFedMeta configs/smoke.yaml
run_one FedMetaEnv configs/smoke.yaml
run_one FedMetaTemporal configs/smoke.yaml

echo ""
echo "[FedMetaBench] All smoke runs completed."
