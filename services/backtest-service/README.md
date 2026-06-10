# Backtest Service

Small internal FastAPI service for historical strategy simulation. It reads
public OHLCV data, runs local backtesting.py simulations, and returns summary
statistics, sampled equity, and trade rows.

Default endpoints:

```text
GET /healthz
GET /strategies
POST /backtest
POST /factor-ic
POST /walk-forward
```

Example:

```text
POST /backtest
{
  "symbol": "BTCUSDT",
  "source": "binance",
  "timeframe": "1d",
  "start": "2023-01-01",
  "end": "2024-01-01",
  "strategy": "ma_cross",
  "params": {"fast": 20, "slow": 50, "trend": 200},
  "cash": 10000,
  "commission": 0.001
}
```

The service is isolated from the execution path and only returns simulation
results.

`POST /factor-ic` is a candidates-only evaluation endpoint. It loads public
OHLCV through the same data adapter, computes generic factor values and future
return labels, then reports Rank IC, Pearson IC, grouped return monotonicity,
factor autocorrelation, and high-correlation redundancy suggestions. It does
not contain strategy rules or any trading/order path.

`POST /walk-forward` is a candidates-only overfit check. It rolls through
train-then-adjacent-test windows, selects strategy parameters only on each
training slice, then reports out-of-sample return, win rate, drawdown, Sharpe,
IS/OOS decay, and data-mining warnings. It does not use in-sample results as
out-of-sample evidence and has no trading/order path.
