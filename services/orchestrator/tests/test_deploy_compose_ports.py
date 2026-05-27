
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "deploy" / "docker-compose.yml"


def test_orchestrator_and_analysis_adapter_bind_host_local_only() -> None:
    text = COMPOSE.read_text()
    assert '"8080:8080"' not in text
    assert '"127.0.0.1:${ORCHESTRATOR_HOST_PORT:-18081}:8080"' in text
    assert '"127.0.0.1:${ANALYSIS_ADAPTER_HOST_PORT:-18085}:8085"' in text


def test_analysis_adapter_uses_internal_tradingagents_bridge() -> None:
    text = COMPOSE.read_text()
    assert 'TA_BRIDGE_URL: ${TA_BRIDGE_URL:-http://tradingagents-bridge:18181}' in text
    assert 'host.docker.internal' not in text


def test_market_data_has_egress_network_without_host_ports() -> None:
    text = COMPOSE.read_text()
    market_section = text.split("\n  market-data:\n", 1)[1].split("\n  analysis-adapter:\n", 1)[0]
    assert "networks: [trading_net, host_access_net]" in market_section
    assert "ports:" not in market_section
