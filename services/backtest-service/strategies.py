from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from backtesting import Strategy  # type: ignore[import-untyped]
from backtesting.lib import crossover  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)
StrategyParams = dict[str, int | float | bool | str]


def sma(values: Any, period: int) -> Any:
    return pd.Series(values).rolling(period).mean().to_numpy()


class MaCrossStrategy(Strategy):  # type: ignore[misc]
    fast = 20
    slow = 50
    trend = 200

    def init(self) -> None:
        self.fast_ma = self.I(sma, self.data.Close, self.fast)
        self.slow_ma = self.I(sma, self.data.Close, self.slow)
        self.trend_ma = self.I(sma, self.data.Close, self.trend)

    def next(self) -> None:
        price = self.data.Close[-1]
        if (
            not self.position
            and price > self.trend_ma[-1]
            and crossover(self.fast_ma, self.slow_ma)
        ):
            self.buy()
        elif self.position and (price < self.trend_ma[-1] or crossover(self.slow_ma, self.fast_ma)):
            self.position.close()


DEFAULT_PARAMS: dict[str, StrategyParams] = {
    "ma_cross": {"fast": 20, "slow": 50, "trend": 200},
}

STRATEGIES: dict[str, type[Strategy]] = {
    "ma_cross": MaCrossStrategy,
}

_BUILTIN_STRATEGIES = frozenset(STRATEGIES)


def register(name: str, cls: type[Strategy], default_params: StrategyParams | None = None) -> bool:
    normalized = name.strip()
    if not normalized:
        logger.warning("Skipping strategy plugin with empty name")
        return False
    if normalized in _BUILTIN_STRATEGIES or normalized in STRATEGIES:
        logger.warning(
            "Skipping strategy plugin %s because the name is already registered", normalized
        )
        return False
    if not isinstance(cls, type) or not issubclass(cls, Strategy):
        logger.warning(
            "Skipping strategy plugin %s because cls is not a Strategy subclass", normalized
        )
        return False
    STRATEGIES[normalized] = cls
    DEFAULT_PARAMS[normalized] = dict(default_params or {})
    logger.info("Loaded strategy plugin %s", normalized)
    return True


def _register_strategy_object(raw: object, source: Path) -> None:
    if not isinstance(raw, dict):
        logger.warning("Skipping strategy plugin %s because STRATEGY is not a dict", source)
        return
    name = raw.get("name")
    cls = raw.get("cls")
    default_params = raw.get("default_params", {})
    if not isinstance(name, str) or not isinstance(default_params, dict):
        logger.warning("Skipping strategy plugin %s because STRATEGY metadata is invalid", source)
        return
    register(name, cls, default_params)  # type: ignore[arg-type]


def load_plugins(directory: str | os.PathLike[str] | None = None) -> None:
    raw_dir = directory if directory is not None else os.getenv("STRATEGY_PLUGINS_DIR", "")
    if not str(raw_dir):
        return
    plugin_dir = Path(raw_dir).expanduser()
    if not plugin_dir.exists() or not plugin_dir.is_dir():
        return
    for path in sorted(plugin_dir.glob("*.py")):
        module_name = f"aegis_strategy_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                logger.warning("Skipping strategy plugin %s because it has no import spec", path)
                continue
            module = importlib.util.module_from_spec(spec)
            module.register = register  # type: ignore[attr-defined]
            spec.loader.exec_module(module)
            if hasattr(module, "STRATEGY"):
                _register_strategy_object(module.STRATEGY, path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping strategy plugin %s after import failure: %s", path, exc)

# TODO: add a strategy adapter for TradingAgents scorecard signal streams.
