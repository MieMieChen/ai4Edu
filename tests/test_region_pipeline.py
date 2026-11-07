import json
from copy import deepcopy

import pytest
from sqlalchemy import select
from test_pipeline import drain
from test_pipeline import runtime as original_runtime

from ai4edu.config import Settings
from ai4edu.db import AnswerRecord, ExamRecord
from ai4edu.execution import execution_settings, freeze_settings
from ai4edu.schemas import PageLayout
from ai4edu.tasks import dispatch_outbox, execute_unit

runtime = original_runtime


def test_padding_profile_is_frozen_and_legacy_profile_keeps_exact_crop():
    settings = Settings(_env_file=None, model="test", region_padding_ratio=0.04)
    frozen = freeze_settings(settings)
    assert frozen["region_padding_ratio"] == 0.04
    assert execution_settings(settings, frozen).region_padding_ratio == 0.04
    frozen.pop("region_padding_ratio")
    assert execution_settings(settings, frozen).region_padding_ratio == 0


def test_preparation_keeps_detected_box_and_uses_neighbor_bounded_padded_box(runtime):
    client, settings, pipeline, provider, asset_id, exam_id = runtime
    settings.region_padding_ratio = 0.04
    db = client.app.state.db
    drain(db, settings, pipeline)
    with db.session() as session:
        exam = session.get(ExamRecord, exam_id)
        second = deepcopy(exam.questions[0])
        second["id"] = "16"
        exam.questions = exam.questions + [second]
        rubric = deepcopy(exam.rubrics["15"])
        rubric["question_id"] = "16"
        exam.rubrics = exam.rubrics | {"16": rubric}
        session.commit()
    original = provider.request

    def request(role, schema, prompt, images):
        if schema is PageLayout:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "rotation": 0,
                                    "regions": [
                                        {
                                            "question_id": "15",
                                            "bbox": [0.1, 0.1, 0.4, 0.9],
                                            "order": 0,
                                        },
                                        {
                                            "question_id": "16",
                                            "bbox": [0.42, 0.1, 0.8, 0.9],
                                            "order": 1,
                                        },
                                    ],
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        return original(role, schema, prompt, images)

    provider.request = request
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam_id,
            "candidates": [{"candidate_id": "anonymous", "kind": "paper", "pages": [asset_id]}],
        },
    )
    assert response.status_code == 202
    pending = []
    dispatch_outbox(db, lambda task, ident: pending.append((task, ident)))
    for task, ident in pending:
        execute_unit(db, settings, pipeline, task, ident)
    with db.session() as session:
        answers = list(
            session.scalars(
                select(AnswerRecord)
                .where(AnswerRecord.job_id == response.json()["id"])
                .order_by(AnswerRecord.question_id)
            )
        )
        first, second = answers[0].assets[0], answers[1].assets[0]
        assert first["detected_bbox"] == [0.1, 0.1, 0.4, 0.9]
        assert first["bbox"][0] == pytest.approx(0.088)
        assert first["bbox"][2] == pytest.approx(0.41)
        assert second["bbox"][0] == pytest.approx(0.41)
        assert first["source_asset_id"] == asset_id
