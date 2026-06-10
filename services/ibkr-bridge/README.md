# IBKR Bridge

Small internal FastAPI wrapper around `ib_async` for Interactive Brokers
Gateway/TWS paper trading. It is intentionally isolated from the shared
`aegis` package so the core repo does not depend on `ib_async`.

Default connection targets paper Gateway on the Docker host:

```text
IBKR_GATEWAY_HOST=host.docker.internal
IBKR_GATEWAY_PORT=4002
IBKR_CLIENT_ID=1
IBKR_CONNECT_TIMEOUT_SEC=10
IBKR_RECONNECT_COOLDOWN_SEC=15
```

`/healthz` reports process health. `/readyz` returns `503` until Gateway/TWS is
reachable. If startup raced Gateway/TWS, `/readyz` and request handlers attempt
a cooldown-limited reconnect before reporting not ready.
