PYTHON_IMAGE ?= python:3.11-slim
DOCKER_RUN = docker run --rm -v "$(PWD)":/work -w /work $(PYTHON_IMAGE) sh -lc

.PHONY: test lint typecheck verify

test:
	$(DOCKER_RUN) "PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1 pip install -q -e .[dev] && pytest -q"

lint:
	$(DOCKER_RUN) "PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1 pip install -q -e .[dev] && ruff check ."

typecheck:
	$(DOCKER_RUN) "PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1 pip install -q -e .[dev] && mypy --strict packages && mypy --strict services/orchestrator/app.py services/orchestrator/db.py && mypy --strict services/risk-engine/app.py && mypy --strict services/execution-service/app.py services/execution-service/db.py && mypy --strict services/market-data/app.py && mypy --strict services/analysis-adapter/app.py services/analysis-adapter/db.py && MYPYPATH=services/ibkr-bridge mypy --strict --explicit-package-bases services/ibkr-bridge/app.py services/ibkr-bridge/ibkr_client.py && MYPYPATH=services/exchange-bridge mypy --strict --explicit-package-bases services/exchange-bridge/app.py services/exchange-bridge/exchange_client.py && MYPYPATH=services/backtest-service mypy --strict --explicit-package-bases services/backtest-service/app.py services/backtest-service/competition.py services/backtest-service/data.py services/backtest-service/factor_ic.py services/backtest-service/funding_arb.py services/backtest-service/healthcheck.py services/backtest-service/strategies.py services/backtest-service/walk_forward.py && MYPYPATH=services/screener-service mypy --strict --explicit-package-bases services/screener-service/app.py services/screener-service/data.py"

verify:
	$(DOCKER_RUN) "PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1 pip install -q -e .[dev] && ruff check . && mypy --strict packages && mypy --strict services/orchestrator/app.py services/orchestrator/db.py && mypy --strict services/risk-engine/app.py && mypy --strict services/execution-service/app.py services/execution-service/db.py && mypy --strict services/market-data/app.py && mypy --strict services/analysis-adapter/app.py services/analysis-adapter/db.py && MYPYPATH=services/ibkr-bridge mypy --strict --explicit-package-bases services/ibkr-bridge/app.py services/ibkr-bridge/ibkr_client.py && MYPYPATH=services/exchange-bridge mypy --strict --explicit-package-bases services/exchange-bridge/app.py services/exchange-bridge/exchange_client.py && MYPYPATH=services/backtest-service mypy --strict --explicit-package-bases services/backtest-service/app.py services/backtest-service/competition.py services/backtest-service/data.py services/backtest-service/factor_ic.py services/backtest-service/funding_arb.py services/backtest-service/healthcheck.py services/backtest-service/strategies.py services/backtest-service/walk_forward.py && MYPYPATH=services/screener-service mypy --strict --explicit-package-bases services/screener-service/app.py services/screener-service/data.py && pytest -q"
