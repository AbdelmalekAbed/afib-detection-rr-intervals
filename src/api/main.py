"""Serve the research API and the built React app on one local origin."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictFloat

from src.api.service import ArtifactError, ResearchService

ROOT = Path(__file__).resolve().parents[2]
Dataset = Literal["afdb", "ltafdb"]
Model = Literal["cnn", "hrv"]
Policy = Literal["fixed", "calibrated"]


@lru_cache(maxsize=1)
def get_service() -> ResearchService:
    return ResearchService(
        Path(os.environ.get("AFIB_CATALOG", ROOT / "artifacts/interface_v1/catalog.json"))
    )


Service = Annotated[ResearchService, Depends(get_service)]


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    rr: Annotated[
        list[Annotated[StrictFloat, Field(ge=0.3, le=2.0)]], Field(min_length=60, max_length=60)
    ]


def create_app(static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="AFib Lab",
        version="1.0.0",
        description="Exploration de recherche, sans usage diagnostique.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )

    @app.exception_handler(ArtifactError)
    async def artifact_error(_request: Request, exc: ArtifactError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def missing_patient(_request: Request, _exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": "Patient ou fenêtre introuvable."})

    @app.get("/api/health")
    def health(service: Service):
        service.summary()
        return {"status": "ok", "version": "optimization_v2"}

    @app.get("/api/summary")
    def summary(service: Service):
        return service.summary()

    @app.get("/api/patients")
    def patients(
        service: Service, dataset: Dataset = "afdb", model: Model = "cnn", policy: Policy = "fixed"
    ):
        return service.patients(dataset, model, policy)

    @app.get("/api/patients/{patient}/trace")
    def trace(
        patient: str,
        service: Service,
        dataset: Dataset = "afdb",
        model: Model = "cnn",
        policy: Policy = "fixed",
        start: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 180,
    ):
        return service.trace(dataset, model, policy, patient, start, limit)

    @app.get("/api/patients/{patient}/window/{index}")
    def window(
        patient: str,
        index: Annotated[int, Field(ge=0)],
        service: Service,
        dataset: Dataset = "afdb",
    ):
        return service.window(dataset, patient, index)

    @app.post(
        "/api/predict",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": PredictionInput.model_json_schema()}},
            }
        },
    )
    async def predict(request: Request, service: Service):
        # Bound the body while reading, including requests without Content-Length.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 8192:
                raise HTTPException(
                    status_code=413, detail="Requête trop volumineuse (8 Kio maximum)."
                )
        try:
            data = PredictionInput.model_validate_json(bytes(body))
        except ValueError:
            raise HTTPException(
                status_code=422, detail="Fournir exactement 60 nombres RR, entre 0,3 et 2 secondes."
            ) from None
        # CPU inference is dispatched off the event loop.
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(service.predict, data.rr)

    @app.get("/api/export/summary")
    def export_summary(service: Service):
        return JSONResponse(
            content=service.summary(),
            headers={"Content-Disposition": 'attachment; filename="afib-lab-resultats.json"'},
        )

    # Unknown API routes must remain JSON errors rather than the SPA shell.
    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
    def unknown_api(rest: str):
        raise HTTPException(status_code=404, detail="Route API introuvable.")

    dist = (static_dir or ROOT / "web/dist").resolve()
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str):
        if path == "favicon.svg" and (dist / path).is_file():
            return FileResponse(dist / path)
        if not (dist / "index.html").is_file():
            raise HTTPException(
                status_code=503, detail="Interface non compilée : exécuter make web-build."
            )
        return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
