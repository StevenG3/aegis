# HypothesisSpec Private File Contract

`scripts/run_hypothesis.py` is the #59 file-only seam from Hermes into the Aegis
backtest core. It reads one local JSON file from the private incubating tree,
validates the discipline flags, calls `aegis.backtest_core.run_backtest`, writes
one private verdict JSON, and appends one private registry row.

The JSON Schema lives at
`packages/aegis/backtest_core/hypothesis_schema.json`.

## Boundary

This public repository contains only the schema, CLI, and synthetic tests. Real
Hermes exports, private strategy notes, evidence, actors, credentials, account
data, raw datasets, registry rows, and verdict outputs remain under:

```text
${AEGIS_STRATEGIES_ROOT}/incubating/olympus59/
```

The seam is offline and human-triggered:

- no network API
- no automatic Hermes pull
- no live trading
- no account, order, wallet, or broker mutation
- no strategy adapter wired in by #59

The #59 CLI therefore returns a validation-only `INSUFFICIENT` verdict until a
future reviewed briefing binds a real local runner.

## Field Mapping

| JSON field | `HypothesisSpec` field | Meaning |
|---|---|---|
| `id` | `key` | Stable private registry id. |
| `type` | `hypothesis_type` | One of `factor`, `combo`, `carry`, `event`, `momentum`, `risk`, `price_action`, `other`. |
| `universe` | `universe` | Predeclared symbols or research universe. |
| `predeclared_signals` | `predeclared_signals` | Signals fixed before evaluation. |
| `params` | `params` | Fixed settings or search grid. |
| `cost_model` | `cost_model` | Fee, slippage, funding/borrow treatment. |
| `benchmark` | `benchmark` | Decision-relevant baseline. |
| `data_source` | `data_source` | Sanitized local dataset label. |
| `trial_n` | `trial_count_n` | All predeclared trials for multiple-testing accounting. |
| `survivor_light` | `survivor_light` | Applies the survivor-light positive-verdict ceiling. |
| `discipline` | `BacktestDiscipline` | Required t+1, locked OOS, walk-forward, full costs, and multiple-testing gates. |
| `trust` | CLI guard metadata | Private registry scope and no-live/read-only assertions. |

`runner` and `verdict_adapter` are intentionally not part of the JSON contract.
They are Python callables and must only be added by reviewed Aegis code.

## Required Gates

`discipline.t_plus_1_execution`, `discipline.locked_oos`,
`discipline.walk_forward`, `discipline.full_costs`, and
`discipline.multiple_testing` must all be `true`.

If `survivor_light` is `true`, `discipline.survivor_ceiling` must also be
`true`; survivor-light specs can never become a robust positive claim through
this seam.

`trust.registry_scope` must be `private`; `trust.predeclared`,
`trust.review_gate`, `trust.no_live`, and `trust.read_only` must be `true`;
`trust.export_contains_private_spec_data` and `trust.live_or_network_required`
must be `false`.

## Private Template

This is a synthetic shape template only:

```json
{
  "id": "placeholder-private-registry-id",
  "type": "combo",
  "universe": ["PLACEHOLDER_SYMBOL"],
  "predeclared_signals": ["placeholder_signal"],
  "params": {
    "placeholder_param": "placeholder_value"
  },
  "cost_model": {
    "fee_bps": 10,
    "slippage_bps": 5,
    "funding_bps_per_period": 0,
    "funding_label": "N/A for spot long-only; perp funding not used"
  },
  "benchmark": "placeholder benchmark",
  "data_source": "sanitized_offline_dataset_label",
  "trial_n": 1,
  "survivor_light": false,
  "trust": {
    "registry_scope": "private",
    "predeclared": true,
    "review_gate": true,
    "export_contains_private_spec_data": false,
    "live_or_network_required": false,
    "no_live": true,
    "read_only": true
  },
  "discipline": {
    "t_plus_1_execution": true,
    "locked_oos": true,
    "walk_forward": true,
    "full_costs": true,
    "multiple_testing": true,
    "survivor_ceiling": false
  }
}
```

## Run Command

```bash
AEGIS_STRATEGIES_ROOT=/home/gggqqy/apps/aegis-strategies \
  python scripts/run_hypothesis.py \
  /home/gggqqy/apps/aegis-strategies/incubating/olympus59/specs/example.json
```

Outputs are written only under
`${AEGIS_STRATEGIES_ROOT}/incubating/olympus59/`:

- `results/<spec-id>-<timestamp>.json`
- `hypothesis_registry.jsonl`

Each registry row records `spec_id`, `trial_n`, verdict, timestamp, spec path,
and result path. The CLI prints `global_trial_n`, the cumulative trial count
across the private registry, so reviewers can decide whether a broader global
FDR/PBO correction is now required.
