from test_pipeline import drain
from test_pipeline import runtime as original_runtime

from ai4edu.config import Settings
from ai4edu.execution import execution_settings, freeze_settings

runtime = original_runtime


def test_upright_question_inputs_go_directly_to_transcription_and_grading(runtime):
    client, settings, pipeline, provider, asset_id, exam_id = runtime
    settings.orient_question_images = False
    settings.verification_policy = "off"
    drain(client.app.state.db, settings, pipeline)
    provider.calls.clear()
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam_id,
            "candidates": [
                {
                    "candidate_id": "normal",
                    "kind": "question",
                    "answers": [{"question_id": "15", "assets": [asset_id]}],
                }
            ],
        },
    )
    assert response.status_code == 202
    drain(client.app.state.db, settings, pipeline)
    assert provider.calls == ["transcribe", "grade"]
    result = client.get(f"/v1/jobs/{response.json()['id']}/results").json()[0]
    assert result["total_score"] == 3
    region = result["answers"][0]["source_regions"][0]
    assert region["rotation"] == 0
    assert region["bbox"] is None
    assert region["source_asset_id"] == asset_id


def test_upright_question_setting_does_not_skip_whole_paper_localization(runtime):
    client, settings, pipeline, provider, asset_id, exam_id = runtime
    settings.orient_question_images = False
    settings.verification_policy = "off"
    drain(client.app.state.db, settings, pipeline)
    provider.calls.clear()
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam_id,
            "candidates": [{"candidate_id": "paper", "kind": "paper", "pages": [asset_id]}],
        },
    )
    drain(client.app.state.db, settings, pipeline)
    assert provider.calls == ["layout", "transcribe", "grade"]
    assert client.get(f"/v1/jobs/{response.json()['id']}").json()["counts"] == {"scored": 1}


def test_orientation_choice_is_frozen_and_legacy_snapshots_keep_orientation():
    settings = Settings(_env_file=None, model="test", orient_question_images=False)
    frozen = freeze_settings(settings)
    assert frozen["orient_question_images"] is False
    assert execution_settings(settings, frozen).orient_question_images is False
    frozen.pop("orient_question_images")
    assert execution_settings(settings, frozen).orient_question_images is True
