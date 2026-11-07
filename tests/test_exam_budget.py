from decimal import Decimal

import pytest
from sqlalchemy import func, select
from test_pipeline import drain, submit
from test_pipeline import runtime as original_runtime

from ai4edu.calls import BudgetExceeded
from ai4edu.db import ExamRecord, ModelCall

runtime = original_runtime


def test_exam_cost_includes_shared_preparation_and_student_calls(runtime):
    client, settings, pipeline, _, _, exam_id = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    submit(runtime)
    drain(db, settings, pipeline)
    with db.session() as session:
        expected = session.scalar(
            select(func.sum(ModelCall.cost)).where(ModelCall.exam_id == exam_id)
        )
        assert session.get(ExamRecord, exam_id).committed_cost == expected
        assert expected > 0


def test_shared_exam_budget_stops_a_new_batch_before_provider_call(runtime):
    client, settings, pipeline, provider, _, exam_id = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    with db.session() as session:
        exam = session.get(ExamRecord, exam_id)
        exam.budget_cny = exam.committed_cost + Decimal("0.00000001")
        previous = exam.committed_cost
        session.commit()
    count = len(provider.calls)
    submit(runtime)
    with pytest.raises(BudgetExceeded, match="试卷"):
        drain(db, settings, pipeline)
    assert len(provider.calls) == count
    with db.session() as session:
        assert session.get(ExamRecord, exam_id).committed_cost == previous


def test_exam_creation_accepts_aggregate_budget(runtime):
    client, _, _, _, _, _ = runtime
    response = client.post(
        "/v1/exams",
        json={
            "title": "bounded experiment",
            "budget_cny": "50",
            "questions": [{"id": "15", "stem": "q", "reference": "a", "max_score": 4}],
        },
    )
    assert response.status_code == 201
    assert client.get(f"/v1/exams/{response.json()['id']}").json()["budget_cny"] == 50


def test_capped_exam_cannot_issue_unpriced_calls(runtime):
    client, settings, pipeline, provider, _, exam_id = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    with db.session() as session:
        session.get(ExamRecord, exam_id).budget_cny = Decimal("50")
        session.commit()
    settings.model_prices = {}
    count = len(provider.calls)
    submit(runtime)
    with pytest.raises(ValueError, match="单价"):
        drain(db, settings, pipeline)
    assert len(provider.calls) == count
