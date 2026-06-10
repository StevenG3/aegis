# Exchange Bridge

Small internal FastAPI wrapper for read-only balances across Binance, OKX, and
Bybit. The bridge uses exchange-native account endpoints so spot/funding,
earn/savings, unified accounts, and futures margin balances are aggregated by
asset before the dashboard renders them.

Default endpoint:

```text
GET /balances
GET /balances?exchange=binance
GET /healthz
GET /readyz
```

Environment variables:

```text
EXCHANGE_API_KEY=<<unset>>
EXCHANGE_API_SECRET=<<unset>>
OKX_API_KEY=<<unset>>
OKX_API_SECRET=<<unset>>
OKX_API_PASSPHRASE=<<unset>>
BYBIT_API_KEY=<<unset>>
BYBIT_API_SECRET=<<unset>>
EXCHANGE_TIMEOUT_SEC=10
EXCHANGE_MIN_USD_DETAIL=10
```

`/healthz` reports process health. `/readyz` returns `503` when no exchange is
configured or none of the configured exchanges can be reached.

`EXCHANGE_MIN_USD_DETAIL` folds per-asset rows whose estimated USD value is
below the threshold into the `OTHER` row. The default is `10`.
