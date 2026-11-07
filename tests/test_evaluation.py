import copy
import importlib
import importlib.util
from decimal import Decimal

import pytest


def _evaluate(predictions, labels):
    assert importlib.util.find_spec("ai4edu.evaluation") is not None, (
        "Denominator-aware offline evaluation is not implemented"
    )
    return importlib.import_module("ai4edu.evaluation").evaluate_predictions(predictions, labels)


def _compare(strategies, labels):
    assert importlib.util.find_spec("ai4edu.evaluation") is not None, (
        "Paired A/B/C comparison is not implemented"
    )
    return importlib.import_module("ai4edu.evaluation").compare_strategies(strategies, labels)


def _label(candidate="candidate_0001", question="15", score="5", maximum="13"):
    return {"candidate_id": candidate, "question_id": question, "score": score, "max_score": maximum}


def _prediction(candidate="candidate_0001", question="15", score="5", status="scored", **extra):
    return {
        "candidate_id": candidate,
        "question_id": question,
        "status": status,
        "score": score,
        **extra,
    }


def test_failed_ungradable_and_missing_predictions_stay_in_coverage_denominator():
    labels = [_label(f"candidate_{i:04}") for i in range(1, 7)]
    predictions = [
        _prediction("candidate_0001", cost_cny="0.10", latency_seconds=1),
        _prediction("candidate_0002", score="6", status="resolved", cost_cny="0.20", latency_seconds=3),
        _prediction("candidate_0003", score=None, status="failed"),
        _prediction("candidate_0004", score=None, status="ungradable"),
        _prediction("candidate_0005", score=None, status="budget_exceeded"),
    ]
    report = _evaluate(predictions, labels)
    assert report["expected_count"] == 6
    assert report["observed_count"] == 5
    assert report["resolved_count"] == 2
    assert report["coverage"] == pytest.approx(2 / 6)
    assert report["status_counts"] == {
        "scored": 1, "resolved": 1, "ungradable": 1, "failed": 1,
        "budget_exceeded": 1, "missing": 1,
    }
    assert report["mae"] == "0.5"
    assert report["exact_rate"] == 0.5
    assert report["within_1_rate"] == report["within_2_rate"] == 1
    assert report["cost"] == {"known_cny": "0.3", "unknown_count": 4}
    assert report["latency_seconds"] == {
        "observed_count": 2, "unknown_count": 4, "p50": 2.0, "p95": 2.9,
    }
    assert len(report["records"]) == 6
    assert report["records"][-1]["status"] == "missing"
    assert report["records"][-1]["score"] is None


def test_error_metrics_use_decimal_arithmetic_and_question_maxima():
    labels = [
        _label("candidate_0001", score="0.2", maximum="1"),
        _label("candidate_0002", question="16", score="5.1", maximum="10"),
        _label("candidate_0003", question="16", score="5.1", maximum="10"),
    ]
    predictions = [
        _prediction("candidate_0001", score="0.3"),
        _prediction("candidate_0002", question="16", score="2.1"),
        _prediction("candidate_0003", question="16", score="7.6"),
    ]
    report = _evaluate(predictions, labels)
    assert Decimal(report["mae"]) == Decimal("5.6") / 3
    assert Decimal(report["nmae"]) == Decimal("0.65") / 3
    assert report["exact_rate"] == 0
    assert report["within_1_rate"] == report["within_2_rate"] == pytest.approx(1 / 3)
    assert report["signed_difference_counts"] == {"-3": 1, "0.1": 1, "2.5": 1}
    assert report["major_error_count"] == 1
    assert report["major_errors"][0]["candidate_id"] == "candidate_0002"
    assert report["major_errors"][0]["difference"] == "-3"
    assert report["per_question"]["16"]["resolved_count"] == 2
    assert report["records"][0]["absolute_error"] == "0.1"


def test_subjective_total_requires_all_five_questions_with_complete_labels():
    labels = [_label(candidate, str(question)) for candidate in ("complete", "incomplete") for question in range(15, 20)]
    labels.append(_label("partial_labels"))
    predictions = [
        _prediction(candidate, str(question), score="6")
        for candidate in ("complete", "incomplete")
        for question in range(15, 20)
        if (candidate, question) != ("incomplete", 19)
    ]
    predictions.append(_prediction("partial_labels"))
    report = _evaluate(predictions, labels)
    totals = report["student_totals"]
    assert totals["expected_count"] == 3
    assert totals["resolved_count"] == 1
    assert totals["coverage"] == pytest.approx(1 / 3)
    assert totals["mae"] == "5"
    records = {row["candidate_id"]: row for row in totals["records"]}
    assert records["complete"]["score"] == "30"
    assert records["complete"]["label_score"] == "25"
    assert records["incomplete"]["score"] is None
    assert records["incomplete"]["partial_score"] == "24"
    assert records["incomplete"]["resolved_question_count"] == 4
    assert records["partial_labels"]["score"] is None


def test_empty_and_all_failed_reports_have_no_fabricated_error_metrics():
    empty = _evaluate([], [])
    assert empty["coverage"] is None
    assert empty["mae"] is None
    assert empty["latency_seconds"]["p95"] is None
    report = _evaluate([_prediction(score=None, status="failed")], [_label()])
    assert report["coverage"] == 0
    assert report["mae"] is None
    assert report["exact_rate"] is None
    assert report["major_error_count"] == 0


def test_student_with_no_resolved_questions_has_no_fabricated_partial_zero():
    labels = [_label(question=str(question)) for question in range(15, 20)]
    report = _evaluate([], labels)
    student = report["student_totals"]["records"][0]
    assert student["score"] is None
    assert student["partial_score"] is None
    assert student["resolved_question_count"] == 0


@pytest.mark.parametrize("problem", ["duplicate_label", "duplicate_prediction", "unknown_candidate", "unknown_question"])
def test_invalid_or_duplicate_prediction_joins_raise(problem):
    labels = [_label()]
    predictions = [_prediction()]
    if problem == "duplicate_label":
        labels.append(_label())
    elif problem == "duplicate_prediction":
        predictions.append(_prediction())
    elif problem == "unknown_candidate":
        predictions[0]["candidate_id"] = "unlabelled"
    else:
        predictions[0]["question_id"] = "not-a-question"
    with pytest.raises(ValueError, match="duplicate|unknown|重复|对应"):
        _evaluate(predictions, labels)


@pytest.mark.parametrize("update", [
    {"score": None}, {"score": "NaN"}, {"score": "13.1"}, {"score": "-0.1"},
    {"score": True}, {"score": "5", "status": "failed"}, {"status": "pending"},
    {"cost_cny": "-1"}, {"cost_cny": "Infinity"}, {"latency_seconds": -1},
    {"latency_seconds": float("nan")}, {"escalated": "false"},
])
def test_invalid_prediction_values_are_not_silently_repaired(update):
    with pytest.raises(ValueError):
        _evaluate([_prediction(**update)], [_label()])


def test_negative_decimal_latency_is_rejected_before_float_underflow():
    with pytest.raises(ValueError, match="latency_seconds"):
        _evaluate([_prediction(latency_seconds="-1e-1000")], [_label()])


def test_paired_comparison_uses_only_common_resolved_units_and_is_order_independent():
    labels = [_label(f"candidate_{i:04}") for i in range(1, 4)]
    a = [_prediction("candidate_0001", score="5"), _prediction("candidate_0002", score="4")]
    b = [_prediction("candidate_0001", score="7"), _prediction("candidate_0003", score="5")]
    report = _compare({"A": a, "B": list(reversed(b))}, labels)
    pair = report["paired"]["A_vs_B"]
    assert pair["common_resolved_count"] == 1
    assert pair["left_mae"] == "0"
    assert pair["right_mae"] == "2"
    assert pair["mae_difference"] == "-2"
    assert pair["score_difference_counts"] == {"-2": 1}
    assert pair["left_better_count"] == 1
    assert report["strategies"]["A"]["resolved_count"] == 2
    assert report["strategies"]["B"]["coverage"] == pytest.approx(2 / 3)
    assert report["c_escalation"] is None


def test_c_diagnostics_measure_error_capture_and_benefit_without_changing_routing():
    labels = [_label(f"candidate_{i:04}") for i in range(1, 6)]
    a = [_prediction(f"candidate_{i:04}") for i in range(1, 6)]
    b = [
        _prediction("candidate_0001", score="1"),
        _prediction("candidate_0002", score="2"),
        _prediction("candidate_0003", score="5"),
        _prediction("candidate_0004", score=None, status="failed"),
        _prediction("candidate_0005", score="5"),
    ]
    c = [
        _prediction("candidate_0001", score="5", escalated=True),
        _prediction("candidate_0002", score="2", escalated=False),
        _prediction("candidate_0003", score="9", escalated=True),
        _prediction("candidate_0004", score="5", escalated=True),
        _prediction("candidate_0005", score="5", escalated=False),
    ]
    original = copy.deepcopy(c)
    report = _compare({"A": a, "B": b, "C": c}, labels)
    diagnostics = report["c_escalation"]
    assert diagnostics["escalated_count"] == 3
    assert diagnostics["b_major_error_count"] == 2
    assert diagnostics["captured_b_major_error_count"] == 1
    assert diagnostics["b_major_error_capture_rate"] == 0.5
    assert diagnostics["captured_b_unresolved_count"] == 1
    assert diagnostics["b_unresolved_rescued_count"] == 1
    assert diagnostics["improved_count"] == 1
    assert diagnostics["worsened_count"] == 1
    assert diagnostics["unchanged_count"] == 0
    assert diagnostics["major_errors_corrected_count"] == 1
    assert diagnostics["major_errors_introduced_count"] == 1
    assert diagnostics["major_errors_retained_count"] == 0
    assert diagnostics["absolute_error_reduction"] == "0"
    assert diagnostics["a_score_match_count"] == 2
    assert diagnostics["a_common_resolved_count"] == 3
    assert c == original

    changed_labels = [{**row, "score": "4"} for row in labels]
    changed = _compare({"A": a, "B": b, "C": c}, changed_labels)
    assert changed["c_escalation"]["escalated_count"] == 3
    assert c == original


def test_extra_runtime_metadata_does_not_enter_report():
    report = _evaluate([_prediction(transcript={"lines": ["untrusted transcript"]}, calls=[{}])], [_label()])
    assert "transcript" not in report["records"][0]
    assert "calls" not in report["records"][0]
