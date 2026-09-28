import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.main import create_app


def test_health_reports_model_loaded() -> None:
    app = create_app(model_loader=lambda _path: object())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_error"] is None


def test_predict_rejects_non_image_upload() -> None:
    app = create_app(model_loader=lambda _path: object())

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"image": ("not-image.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]
