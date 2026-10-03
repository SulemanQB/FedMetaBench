# CPU smoke tests for all FedMetaBench benchmarks.
[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

& $Python -c "import sys; print('[FedMetaBench] Using:', sys.executable, sys.version.split()[0])"

$benchmarks = @(
    @{ Name = "FedNoisyMAML"; Config = "configs/smoke.yaml" },
    @{ Name = "FairFedMeta"; Config = "configs/smoke.yaml" },
    @{ Name = "FedMetaEnv"; Config = "configs/smoke.yaml" },
    @{ Name = "FedMetaTemporal"; Config = "configs/smoke.yaml" }
)

foreach ($benchmark in $benchmarks) {
    Write-Host ""
    Write-Host "=== Smoke: $($benchmark.Name) ==="
    Push-Location (Join-Path $Root "benchmarks/$($benchmark.Name)")
    try {
        & $Python run_experiment.py --config $benchmark.Config
        if ($LASTEXITCODE -ne 0) {
            throw "Smoke run failed for $($benchmark.Name) with exit code $LASTEXITCODE."
        }
        Write-Host "OK: $($benchmark.Name)"
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "[FedMetaBench] All smoke runs completed."
