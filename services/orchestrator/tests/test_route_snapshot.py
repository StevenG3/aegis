import importlib
import sys
from pathlib import Path

EXPECTED_ROUTES = [
    (["GET"], "/healthz", "healthz", "dict"),
    (["GET"], "/readyz", "readyz", "dict"),
    (["POST"], "/notifications/subscribe", "subscribe_notifications", None),
    (["POST"], "/notifications/unsubscribe", "unsubscribe_notifications", None),
    (["GET"], "/notifications/deliveries", "list_notification_deliveries", None),
    (["GET"], "/ev-shadow/report", "get_ev_shadow_report", None),
    (["POST"], "/calibration/recompute", "recompute_calibration", None),
    (["GET"], "/calibration", "get_calibration", None),
    (["POST"], "/memory/record", "record_memory_endpoint", None),
    (["GET"], "/memory/recall", "recall_memory", None),
    (["GET"], "/memory/lineage/{memory_id}", "memory_lineage", None),
    (["POST"], "/autonomy/settings", "update_autonomy", None),
    (["GET"], "/autonomy/settings", "get_autonomy_settings", None),
    (["GET"], "/autonomy/today", "get_autonomy_today", None),
    (["POST"], "/admin/live-autonomy/disable", "disable_live_autonomy", None),
    (["POST"], "/admin/live-autonomy/enable", "reenable_live_autonomy", None),
    (["POST"], "/live-autonomy/settings", "update_live_autonomy", None),
    (["GET"], "/live-autonomy/settings", "get_live_autonomy_settings", None),
    (["GET"], "/live-autonomy/today", "get_live_autonomy_today", None),
    (["GET"], "/paper/autonomy/observability", "get_paper_autonomy_observability", None),
    (["POST"], "/paper/autonomy/alerts", "post_paper_autonomy_alerts", None),
    (["POST"], "/paper/autonomy/digest", "post_paper_autonomy_digest", None),
    (["GET"], "/paper/autonomy/readiness", "get_paper_autonomy_readiness", None),
    (["GET"], "/paper/bootstrap/status", "get_paper_bootstrap_status", None),
    (
        ["POST"],
        "/paper/bootstrap/evaluate-guardrails",
        "post_paper_bootstrap_evaluate_guardrails",
        None,
    ),
    (["POST"], "/paper/bootstrap/resume", "post_paper_bootstrap_resume", None),
    (["POST"], "/watchlist", "add_watchlist", None),
    (["GET"], "/watchlist", "list_watchlist", None),
    (["DELETE"], "/watchlist/{symbol}", "delete_watchlist", None),
    (["POST"], "/admin/live-unlock", "issue_live_unlock", None),
    (["POST"], "/scorecards", "create_scorecard", None),
    (["GET"], "/scorecards/{scorecard_id}", "get_scorecard", None),
    (["GET"], "/scorecards", "list_scorecards", None),
    (["GET"], "/scorecard-outcomes", "list_outcomes", None),
    (["GET"], "/scorecard-outcomes/summary", "outcomes_summary", None),
    (["GET"], "/factor-attribution", "factor_attribution", None),
    (["POST"], "/reflect/pending", "reflect_pending", None),
    (["GET"], "/scorecard-outcomes/{outcome_id}", "get_outcome", None),
    (["POST"], "/scorecard-outcomes/{outcome_id}/trailing", "update_outcome_trailing", None),
    (["GET"], "/pnl/today", "get_pnl_today", None),
    (["POST"], "/intents/from_nl", "create_intent_from_nl", None),
    (["POST"], "/intents/from_scorecard", "create_intent_from_scorecard", None),
    (["POST"], "/intents", "create_intent", None),
    (["GET"], "/intents", "list_intents", "dict"),
    (["GET"], "/exposure", "get_exposure", "dict"),
    (["POST"], "/intents/{intent_id}/confirm", "confirm_intent", None),
    (["DELETE"], "/intents/{intent_id}", "cancel_intent", None),
    (["GET"], "/paper/positions", "get_paper_positions", None),
    (["POST"], "/intents/{intent_id}/refresh", "refresh_intent", None),
    (["GET"], "/intents/{intent_id}", "get_intent", None),
    (["POST"], "/reconcile/ibkr/apply", "post_reconcile_apply", "ReconcileApplyResponse"),
    (["POST"], "/reconcile/ibkr", "trigger_ibkr_reconcile", None),
    (["GET"], "/reconcile/ibkr/latest", "get_latest_ibkr_reconcile", None),
]


def test_orchestrator_package_import_and_route_snapshot_are_stable() -> None:
    services_dir = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(services_dir))

    app_module = importlib.import_module("orchestrator.app")
    snapshot_module = importlib.import_module("orchestrator.route_snapshot")
    snapshot = snapshot_module.application_route_snapshot(app_module.app)

    assert app_module.app.title == "orchestrator"
    assert len(snapshot) == 53
    assert [
        (
            route["methods"],
            route["path"],
            route["name"],
            route["response_model"],
        )
        for route in snapshot
    ] == EXPECTED_ROUTES
