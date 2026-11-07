"""Label-isolated, denominator-aware offline score evaluation.

Scores, errors and costs use Decimal arithmetic and JSON-compatible decimal
strings. Rates use their stated denominator; undefined rates/errors are null.
``scored`` and ``resolved`` are both resolved prediction states. Missing records
remain in the expected denominator and have unknown cost/latency. Latency P50
and P95 use linear interpolation at (n-1)*p among known unit latencies.

This module never creates predictions or decides escalation. C diagnostics read
the already-recorded runtime ``escalated`` flag; labels are diagnostic only.
"""

import math
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from itertools import combinations

STATUSES = ("scored", "resolved", "ungradable", "failed", "budget_exceeded", "missing")
RESOLVED_STATUSES = {"scored", "resolved"}
SUBJECTIVE_QUESTIONS = {"15", "16", "17", "18", "19"}


def _decimal(value, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, str | int | float | Decimal):
        raise ValueError(f"invalid {field}: expected a finite decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"invalid {field}: expected a finite decimal") from None
    if not result.is_finite():
        raise ValueError(f"invalid {field}: expected a finite decimal")
    return result


def _string(value: Decimal) -> str:
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return "0" if result == "-0" else result


def _mean(values: list[Decimal]) -> str | None:
    return _string(sum(values, Decimal(0)) / len(values)) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _key(row: dict) -> tuple[str, str]:
    identifiers = row.get("candidate_id"), row.get("question_id")
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("candidate_id and question_id must be nonempty strings")
    return identifiers


def _label_index(labels: list[dict]) -> dict:
    indexed, maxima = {}, {}
    for label in labels:
        key = _key(label)
        if key in indexed:
            raise ValueError("duplicate label candidate/question join")
        score = _decimal(label.get("score"), "label score")
        maximum = _decimal(label.get("max_score"), "max_score")
        if maximum <= 0 or not 0 <= score <= maximum:
            raise ValueError("label score/max_score is outside its legal range")
        if key[1] in maxima and maxima[key[1]] != maximum:
            raise ValueError("inconsistent max_score for the same question")
        maxima[key[1]] = maximum
        indexed[key] = {"score": score, "max_score": maximum}
    return indexed


def _normalized_records(predictions: list[dict], labels: dict) -> list[dict]:
    indexed = {}
    for prediction in predictions:
        key = _key(prediction)
        if key in indexed:
            raise ValueError("duplicate prediction candidate/question join")
        if key not in labels:
            raise ValueError("unknown prediction candidate/question join")
        indexed[key] = prediction
    records = []
    for key, label in sorted(labels.items()):
        prediction = indexed.get(key, {"status": "missing", "score": None})
        status = prediction.get("status")
        if status not in STATUSES:
            raise ValueError("invalid prediction status")
        score = prediction.get("score")
        if status in RESOLVED_STATUSES:
            score = _decimal(score, "prediction score")
            if not 0 <= score <= label["max_score"]:
                raise ValueError("prediction score is outside its legal range")
        elif score is not None:
            raise ValueError("unresolved prediction score must be null")
        cost = prediction.get("cost_cny")
        if cost is not None:
            cost = _decimal(cost, "cost_cny")
            if cost < 0:
                raise ValueError("cost_cny must be nonnegative")
        latency = prediction.get("latency_seconds")
        if latency is not None:
            latency = _decimal(latency, "latency_seconds")
            if latency < 0:
                raise ValueError("latency_seconds must be finite and nonnegative")
            latency = float(latency)
            if not math.isfinite(latency):
                raise ValueError("latency_seconds must be finite and nonnegative")
        escalated = prediction.get("escalated", False)
        if not isinstance(escalated, bool):
            raise ValueError("escalated must be a boolean")
        difference = score - label["score"] if score is not None else None
        records.append({
            "candidate_id": key[0], "question_id": key[1], "status": status,
            "score": _string(score) if score is not None else None,
            "label_score": _string(label["score"]), "max_score": _string(label["max_score"]),
            "difference": _string(difference) if difference is not None else None,
            "absolute_error": _string(abs(difference)) if difference is not None else None,
            "cost_cny": _string(cost) if cost is not None else None,
            "latency_seconds": latency, "escalated": escalated,
        })
    return records


def _distribution(values: list[Decimal]) -> dict[str, int]:
    counts = Counter(values)
    return {_string(value): counts[value] for value in sorted(counts)}


def _error_metrics(records: list[dict]) -> dict:
    resolved = [row for row in records if row["score"] is not None]
    differences = [Decimal(row["difference"]) for row in resolved]
    errors = [abs(value) for value in differences]
    return {
        "mae": _mean(errors),
        "exact_rate": _rate(sum(error == 0 for error in errors), len(errors)),
        "within_1_rate": _rate(sum(error <= 1 for error in errors), len(errors)),
        "within_2_rate": _rate(sum(error <= 2 for error in errors), len(errors)),
        "nmae": _mean([
            Decimal(row["absolute_error"]) / Decimal(row["max_score"]) for row in resolved
        ]),
        "signed_difference_counts": _distribution(differences),
        "major_error_count": sum(error >= 3 for error in errors),
        "major_errors": [row for row in resolved if Decimal(row["absolute_error"]) >= 3],
    }


def _percentile(values: list[Decimal], percentile: Decimal) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _summary(records: list[dict]) -> dict:
    counts = Counter(row["status"] for row in records)
    resolved_count = sum(counts[status] for status in RESOLVED_STATUSES)
    known_costs = [Decimal(row["cost_cny"]) for row in records if row["cost_cny"] is not None]
    latencies = [Decimal(str(row["latency_seconds"])) for row in records if row["latency_seconds"] is not None]
    escalated_count = sum(row["escalated"] for row in records)
    return {
        "expected_count": len(records), "observed_count": len(records) - counts["missing"],
        "resolved_count": resolved_count, "coverage": _rate(resolved_count, len(records)),
        "status_counts": {status: counts[status] for status in STATUSES},
        **_error_metrics(records),
        "cost": {
            "known_cny": _string(sum(known_costs, Decimal(0))),
            "unknown_count": len(records) - len(known_costs),
        },
        "latency_seconds": {
            "observed_count": len(latencies), "unknown_count": len(records) - len(latencies),
            "p50": _percentile(latencies, Decimal("0.5")),
            "p95": _percentile(latencies, Decimal("0.95")),
        },
        "escalated_count": escalated_count, "escalation_rate": _rate(escalated_count, len(records)),
    }


def _student_totals(records: list[dict]) -> dict:
    candidates = defaultdict(list)
    for row in records:
        candidates[row["candidate_id"]].append(row)
    totals = []
    for candidate, questions in sorted(candidates.items()):
        subjective = [row for row in questions if row["question_id"] in SUBJECTIVE_QUESTIONS]
        resolved = [row for row in subjective if row["score"] is not None]
        all_labels = {row["question_id"] for row in questions} == SUBJECTIVE_QUESTIONS
        complete = all_labels and len(resolved) == 5
        partial_score = sum((Decimal(row["score"]) for row in resolved), Decimal(0))
        label_score = sum((Decimal(row["label_score"]) for row in subjective), Decimal(0))
        maximum = sum((Decimal(row["max_score"]) for row in subjective), Decimal(0))
        difference = partial_score - label_score if complete else None
        totals.append({
            "candidate_id": candidate, "status": "resolved" if complete else "incomplete",
            "expected_question_count": 5, "label_question_count": len(subjective),
            "resolved_question_count": len(resolved),
            "partial_score": _string(partial_score) if resolved else None,
            "score": _string(partial_score) if complete else None,
            "label_score": _string(label_score) if all_labels else None,
            "max_score": _string(maximum) if all_labels else None,
            "difference": _string(difference) if difference is not None else None,
            "absolute_error": _string(abs(difference)) if difference is not None else None,
        })
    resolved_count = sum(row["score"] is not None for row in totals)
    return {
        "expected_count": len(totals), "resolved_count": resolved_count,
        "coverage": _rate(resolved_count, len(totals)),
        **_error_metrics(totals), "records": totals,
    }


def evaluate_predictions(predictions: list[dict], labels: list[dict]) -> dict:
    """Evaluate normalized terminal predictions against an exact offline join.

    Duplicate or unknown candidate/question keys, nonfinite/out-of-range values,
    and nonnull unresolved scores raise ValueError. Runtime metadata extras are
    ignored. Question error rates use resolved units; coverage, missing costs
    and missing latencies use all expected labels. Complete subjective totals
    require every question 15--19, with all five predictions resolved.
    """
    records = _normalized_records(predictions, _label_index(labels))
    questions = defaultdict(list)
    for record in records:
        questions[record["question_id"]].append(record)
    return {
        **_summary(records),
        "per_question": {qid: _summary(rows) for qid, rows in sorted(questions.items())},
        "student_totals": _student_totals(records), "records": records,
    }


def _paired_metrics(pairs: list[tuple[dict, dict]], expected_count: int) -> dict:
    common = [(left, right) for left, right in pairs if left["score"] is not None and right["score"] is not None]
    left_errors = [Decimal(left["absolute_error"]) for left, _ in common]
    right_errors = [Decimal(right["absolute_error"]) for _, right in common]
    error_differences = [left - right for left, right in zip(left_errors, right_errors, strict=True)]
    score_differences = [Decimal(left["score"]) - Decimal(right["score"]) for left, right in common]
    return {
        "expected_count": expected_count, "common_resolved_count": len(common),
        "common_resolved_coverage": _rate(len(common), expected_count),
        "left_mae": _mean(left_errors), "right_mae": _mean(right_errors),
        "mae_difference": _mean(error_differences),
        "left_better_count": sum(value < 0 for value in error_differences),
        "right_better_count": sum(value > 0 for value in error_differences),
        "equal_error_count": sum(value == 0 for value in error_differences),
        "mean_score_difference": _mean(score_differences),
        "score_difference_counts": _distribution(score_differences),
    }


def _paired_report(left_name: str, right_name: str, left: dict, right: dict) -> dict:
    right_units = {_key(row): row for row in right["records"]}
    right_students = {row["candidate_id"]: row for row in right["student_totals"]["records"]}
    return {
        "left": left_name, "right": right_name,
        **_paired_metrics([(row, right_units[_key(row)]) for row in left["records"]], left["expected_count"]),
        "student_totals": _paired_metrics([
            (row, right_students[row["candidate_id"]]) for row in left["student_totals"]["records"]
        ], left["student_totals"]["expected_count"]),
    }


def _c_diagnostics(reports: dict[str, dict]) -> dict | None:
    if "B" not in reports or "C" not in reports:
        return None
    b = {_key(row): row for row in reports["B"]["records"]}
    c = {_key(row): row for row in reports["C"]["records"]}
    a = {_key(row): row for row in reports["A"]["records"]} if "A" in reports else {}
    escalated = {key for key, row in c.items() if row["escalated"]}
    b_major = {key for key, row in b.items() if row["absolute_error"] is not None and Decimal(row["absolute_error"]) >= 3}
    b_unresolved = {key for key, row in b.items() if row["score"] is None}
    common = {key for key in escalated if b[key]["score"] is not None and c[key]["score"] is not None}
    reductions = [Decimal(b[key]["absolute_error"]) - Decimal(c[key]["absolute_error"]) for key in sorted(common)]
    corrected = sum(Decimal(b[key]["absolute_error"]) >= 3 and Decimal(c[key]["absolute_error"]) < 3 for key in common)
    introduced = sum(Decimal(b[key]["absolute_error"]) < 3 and Decimal(c[key]["absolute_error"]) >= 3 for key in common)
    retained = sum(Decimal(b[key]["absolute_error"]) >= 3 and Decimal(c[key]["absolute_error"]) >= 3 for key in common)
    a_common = {key for key in escalated if key in a and a[key]["score"] is not None and c[key]["score"] is not None}
    a_matches = sum(Decimal(a[key]["score"]) == Decimal(c[key]["score"]) for key in a_common)
    return {
        "expected_count": len(c), "escalated_count": len(escalated),
        "escalation_rate": _rate(len(escalated), len(c)),
        "b_major_error_count": len(b_major),
        "captured_b_major_error_count": len(b_major & escalated),
        "b_major_error_capture_rate": _rate(len(b_major & escalated), len(b_major)),
        "escalation_precision_for_b_major_errors": _rate(len(b_major & escalated), len(escalated)),
        "b_unresolved_count": len(b_unresolved),
        "captured_b_unresolved_count": len(b_unresolved & escalated),
        "b_unresolved_capture_rate": _rate(len(b_unresolved & escalated), len(b_unresolved)),
        "escalated_common_resolved_count": len(common),
        "improved_count": sum(value > 0 for value in reductions),
        "worsened_count": sum(value < 0 for value in reductions),
        "unchanged_count": sum(value == 0 for value in reductions),
        "absolute_error_reduction": _string(sum(reductions, Decimal(0))) if reductions else None,
        "mean_absolute_error_reduction": _mean(reductions),
        "major_errors_corrected_count": corrected, "major_errors_introduced_count": introduced,
        "major_errors_retained_count": retained,
        "b_unresolved_rescued_count": sum(b[key]["score"] is None and c[key]["score"] is not None for key in escalated),
        "b_resolved_lost_count": sum(b[key]["score"] is not None and c[key]["score"] is None for key in escalated),
        "c_remaining_major_error_count": reports["C"]["major_error_count"],
        "a_available": "A" in reports, "a_common_resolved_count": len(a_common),
        "a_score_match_count": a_matches, "a_score_match_rate": _rate(a_matches, len(a_common)),
        "a_mean_absolute_error_difference": _mean([
            Decimal(c[key]["absolute_error"]) - Decimal(a[key]["absolute_error"]) for key in sorted(a_common)
        ]),
    }


def compare_strategies(strategies: dict[str, list[dict]], labels: list[dict]) -> dict:
    """Compare every pair on common resolved units and complete student totals.

    In ``paired``, mae_difference and score_difference are left minus right.
    C benefit counts apply only to flagged escalations with both B/C scores;
    rescues/lost coverage are reported separately. Error capture uses >=3 points.
    A agreement is evaluated on flagged C units where both A/C resolved.
    No routing rule or choice of model receives labels from this function.
    """
    reports = {name: evaluate_predictions(predictions, labels) for name, predictions in sorted(strategies.items())}
    return {
        "strategies": reports,
        "paired": {
            f"{left}_vs_{right}": _paired_report(left, right, reports[left], reports[right])
            for left, right in combinations(reports, 2)
        },
        "c_escalation": _c_diagnostics(reports),
    }
