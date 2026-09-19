from model_policy import evaluate_model_policy


def make_report(status):
    return {
        "model_version": "rf-2024-v1.0.0",
        "overall_status": status,
        "data_drift": {
            "status": "stable",
        },
        "performance": {
            "status": status,
        },
        "alerts": [],
    }


def test_stable_policy_continues():
    policy = evaluate_model_policy(
        make_report("stable")
    )

    assert policy["decision"] == "continue"
    assert policy["allow_current_model"] is True
    assert policy["manual_review_required"] is False
    assert policy["automatic_retraining"] is False


def test_watch_policy_continues_with_monitoring():
    policy = evaluate_model_policy(
        make_report("watch")
    )

    assert (
        policy["decision"]
        == "continue_with_monitoring"
    )

    assert policy["allow_current_model"] is True
    assert policy["manual_review_required"] is False
    assert policy["automatic_model_replacement"] is False


def test_high_degradation_requires_manual_review():
    policy = evaluate_model_policy(
        make_report("high_degradation")
    )

    assert (
        policy["decision"]
        == "manual_review_required"
    )

    assert policy["manual_review_required"] is True
    assert policy["automatic_retraining"] is False
    assert policy["automatic_model_replacement"] is False


def test_unknown_status_blocks_automatic_decision():
    policy = evaluate_model_policy(
        make_report("unknown")
    )

    assert (
        policy["decision"]
        == "manual_review_required"
    )

    assert policy["allow_current_model"] is False
    assert policy["manual_review_required"] is True
