# Screener Service

Internal FastAPI service for valuation screening with public yfinance data.
It returns watchlist candidates and sector valuation summaries for further
research. It is not a buy signal and is isolated from trading workflows.

Default endpoints:

```text
GET /healthz
GET /readyz
GET /sectors
POST /screen
```

Example:

```text
POST /screen
{
  "universe": ["AAPL", "MSFT", "JPM"],
  "filters": {"max_pe": 25},
  "sort_by": "trailing_pe",
  "limit": 10
}
```

The response always includes:

```text
"disclaimer": "valuation screen, candidates only, not a buy signal"
```
