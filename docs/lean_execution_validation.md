# LEAN Execution Validation Seam

This seam lets Aegis ingest a private QuantConnect LEAN backtest or paper report as
an execution-stage validation artifact. LEAN is not part of Aegis and does not make
the research verdict. Aegis remains the strategy judgment layer; LEAN is an execution
simulator whose report can unlock or reject execution realism.

## Boundary

- Offline/file-only in this public repository.
- No live trading, broker login, wallet, account API, or order submission.
- Spec, raw LEAN report, and validation result must live under
  `${AEGIS_STRATEGIES_ROOT}/incubating/<task>/`.
- Public code contains only the schema/validator/CLI and synthetic tests.

## Flow

```text
Aegis EDGE/SUGGESTIVE research verdict
  -> private LEAN execution spec
  -> LEAN backtest or paper report generated outside Aegis
  -> scripts/lean_execution_validation.py
  -> Aegis execution verdict
```

## Minimal Private Spec

```json
{
  "id": "lean-spx-vrp-unit",
  "source_hypothesis_id": "olympus85c_spx_vrp",
  "source_aegis_state": "EDGE",
  "engine": "lean",
  "mode": "backtest",
  "live_trading": false,
  "read_only": true,
  "data_adequacy": "limited",
  "unlock_condition": "broker-native paper fills and paid PIT option chains"
}
```

## Minimal LEAN Report

```json
{
  "engine": "lean",
  "live_trading": false,
  "executable_fills": false,
  "order_count": 42,
  "total_fees": 123.0,
  "total_slippage": 45.0,
  "metrics": {
    "annualized_return": 0.08,
    "sharpe": 1.2,
    "max_drawdown": -0.12
  },
  "benchmark": {
    "annualized_return": 0.05,
    "sharpe": 0.6
  }
}
```

## Run

```bash
AEGIS_STRATEGIES_ROOT=/path/to/private/aegis-strategies \
python scripts/lean_execution_validation.py \
  /path/to/private/aegis-strategies/incubating/olympus85/lean-spec.json \
  /path/to/private/aegis-strategies/incubating/olympus85/lean-report.json
```

The result is written back under the same private incubating task. The first
verdict gate is intentionally conservative:

- `EXECUTION_VALID_PENDING_FINAL_GATE`: costed report beats benchmark on return
  and Sharpe, has orders, and passes the tail gate.
- `EXECUTION_FAIL`: report is valid but does not beat benchmark, lacks cost/order
  evidence, or breaches the tail gate.
- `INSUFFICIENT`: spec/report is malformed, live trading is present, or required
  execution evidence is missing.
