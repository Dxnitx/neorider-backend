import os

os.environ.setdefault("TESTING", "true")

from fastapi.testclient import TestClient

from app.main import app
from app.services.mmc1_model_service import (
    CLASS_MAPPING_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    SCALER_PATH,
)


def test_app_model_status_and_required_artifacts():
    for artifact in (
        MODEL_PATH,
        SCALER_PATH,
        FEATURE_COLUMNS_PATH,
        CLASS_MAPPING_PATH,
    ):
        assert artifact.is_file(), f"Missing ML artifact: {artifact}"

    with TestClient(app) as client:
        response = client.get("/ml/model-status")

    assert response.status_code == 200
    status = response.json()
    assert status["loaded"] is True
    assert status["features"] == 112
    assert status["model_features"] == 112
    assert status["scaler_features"] == 112
