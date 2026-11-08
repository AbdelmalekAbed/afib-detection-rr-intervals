"""Web contracts and real-candidate checks; scientific env can skip web-only tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api.main import create_app, get_service
from src.api.service import ArtifactError, ResearchService

CATALOG = Path(__file__).resolve().parents[1] / "artifacts/interface_v1/catalog.json"


@pytest.fixture(scope="module")
def service():
    if not CATALOG.is_file():
        pytest.skip("Local v2 candidate is not distributed with Git; run make web-prepare.")
    return ResearchService(CATALOG)


@pytest.fixture
def client(service, tmp_path):
    app = create_app(tmp_path)
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as client:
        yield client


def test_summary_matches_frozen_metrics(client, service):
    data = client.get("/api/summary").json()
    assert data["datasets"]["afdb"]["n_patients"] == 25
    assert data["datasets"]["ltafdb"]["n_patients"] == 84
    for dataset, folder in (("afdb", "cnn_lstm"),):
        reference = json.loads(
            (service.root / "optimization_v2" / folder / "metrics.json").read_text()
        )
        assert data["datasets"][dataset]["models"]["cnn"]["fixed"] == reference["metrics"]
        assert (
            data["datasets"][dataset]["models"]["cnn"]["calibrated"]
            == reference["calibrated_metrics"]
        )


@pytest.mark.parametrize("dataset", ["afdb", "ltafdb"])
@pytest.mark.parametrize("model", ["cnn", "hrv"])
@pytest.mark.parametrize("policy", ["fixed", "calibrated"])
def test_patient_aggregation_and_pagination(client, service, dataset, model, policy):
    query = f"dataset={dataset}&model={model}&policy={policy}"
    patients = client.get(f"/api/patients?{query}").json()
    expected = service.catalog["datasets"][dataset]["models"][model][policy]
    scored = [p["f1"] for p in patients if p["scorable"]]
    assert len(scored) == expected["n_patients_scored"]
    assert np.mean(scored) == pytest.approx(expected["mean_patient_f1"])
    patient = patients[0]["id"]
    first = client.get(f"/api/patients/{patient}/trace?{query}&start=0&limit=5").json()
    second = client.get(f"/api/patients/{patient}/trace?{query}&start=5&limit=5").json()
    assert [p["index"] for p in first["points"] + second["points"]] == list(range(10))
    data = service.scores(dataset, model)
    mask = data["groups"] == patient
    assert [p["label"] for p in first["points"]] == data["y"][mask][:5].tolist()
    assert [p["score"] for p in second["points"]] == data["scores"][mask][5:10].tolist()
    assert (
        client.get(f"/api/patients/{patient}/trace?{query}&start={first['total']}&limit=5").json()[
            "points"
        ]
        == []
    )


@pytest.mark.parametrize(
    "rr",
    [[0.8] * 59, [0.8] * 61, [True] * 60, [".8"] * 60, [0.2] * 60, [2.1] * 60, [float("nan")] * 60],
)
def test_reject_invalid_rr(client, rr):
    # allow_nan=True intentionally sends a nonstandard NaN token to the validator.
    response = client.post(
        "/api/predict", content=json.dumps({"rr": rr}), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_rr_request_size_and_unknown_fields(client):
    assert client.post("/api/predict", content=" " * 8193).status_code == 413
    assert (
        client.post(
            "/api/predict", json={"rr": [0.8] * 60, "checkpoint": "/tmp/custom.pt"}
        ).status_code
        == 422
    )


def test_actual_external_window_reproduces_frozen_ensemble(client, service):
    patients = service.patients("ltafdb", "cnn", "fixed")
    patient = patients[0]["id"]
    rr = client.get(f"/api/patients/{patient}/window/0?dataset=ltafdb").json()["rr"]
    response = client.post("/api/predict", json={"rr": rr})
    assert response.status_code == 200
    result = response.json()
    data = service.scores("ltafdb", "cnn")
    idx = np.flatnonzero(data["groups"] == patient)[0]
    assert result["score"] == pytest.approx(float(data["scores"][idx]), abs=1e-5)
    assert result["calibrated_vote"] == int(data["calibrated_decisions"][idx])
    assert len(result["members"]) == 5
    assert [r["threshold"] for r in result["members"]] == [
        f["threshold"] for f in service.catalog["folds"]
    ]


def test_invalid_resources_and_bounds(client):
    assert client.get("/api/patients?dataset=unknown").status_code == 422
    assert client.get("/api/patients?model=../../etc/passwd").status_code == 422
    assert client.get("/api/patients/absent/trace").status_code == 404
    assert client.get("/api/patients/absent/window/0").status_code == 404
    assert client.get("/api/patients/04015/trace?limit=100000").status_code == 422
    assert client.get("/api/patients/04015/trace?start=-1").status_code == 422
    assert client.get("/api/unknown").status_code == 404
    assert client.get("/api/health").status_code == 200


def test_missing_artifacts_are_a_service_error(tmp_path):
    app = create_app(tmp_path)

    def unavailable():
        return ResearchService(tmp_path / "missing.json")

    app.dependency_overrides[get_service] = unavailable
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 503
        assert client.get("/").status_code == 503


def test_integrity_fails_closed(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "scores.npz").write_bytes(b"modified")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_root": str(root),
                "references": {"scores.npz": "0" * 64},
            }
        )
    )
    service = ResearchService(catalog)
    with pytest.raises(ArtifactError, match="empreinte"):
        service.path("scores.npz")
    with pytest.raises(ArtifactError, match="référencé"):
        service.path("../elsewhere.npz")


def test_catalog_preparation_reproduces_verified_references(service, tmp_path):
    from scripts.prepare_interface import prepare

    output = tmp_path / "catalog.json"
    catalog = prepare(service.root / "optimization_v2", output)
    assert catalog["references"] == service.catalog["references"]
    assert catalog["datasets"] == service.catalog["datasets"]
    assert catalog["folds"] == service.catalog["folds"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(service.root / "optimization_v2", output)
    assert output.read_bytes() == before


def test_prediction_schema_documents_sixty_bounded_rr(client):
    schema = client.get("/api/openapi.json").json()
    rr = schema["paths"]["/api/predict"]["post"]["requestBody"]["content"]["application/json"][
        "schema"
    ]["properties"]["rr"]
    assert rr["minItems"] == rr["maxItems"] == 60
    assert rr["items"]["minimum"] == 0.3
    assert rr["items"]["maximum"] == 2.0
