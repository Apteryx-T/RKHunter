from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from . import DATABASE_SCHEMA_VERSION, TOOL_VERSION
from .service import AnnotationService, RevisionConflict
from .adapters import default_registry

API_VERSION = "v1"


class ProjectCreate(BaseModel):
    id: str
    name: str
    dataset_root: str
    image_dir: str = "images"
    label_dir: str = "labels"
    classes: dict[int, str]


class ImportRequest(BaseModel):
    imported_status: str = "auto_labeled"


class AnnotationValue(BaseModel):
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = None
    source: str = "manual"
    model_id: str | None = None
    model_revision_id: str | None = None
    warning: str | None = None


class AnnotationSave(BaseModel):
    expected_revision: int = Field(ge=0)
    status: str = "reviewed"
    actor: str = "local-user"
    annotations: list[AnnotationValue]


class ModelCreate(BaseModel):
    id: str
    name: str
    adapter: str = "ultralytics_yolo"
    weights_path: str
    config: dict[str, Any] = Field(default_factory=dict)


class AutoLabelRequest(BaseModel):
    model_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    replace_auto: bool = True


class RunCreate(BaseModel):
    project_id: str
    model_id: str
    image_ids: list[str] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    output_root: str = "experiments/annotation-tool/exports"
    reviewed_only: bool = True
    copy_mode: str = "copy"


class JobManager:
    def __init__(self, service: AnnotationService):
        self.service = service
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rkhunter-autolabel")
        self._submitted: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._submitted:
                return
            self._submitted.add(run_id)
        self.executor.submit(self._execute, run_id)

    def _execute(self, run_id: str) -> None:
        errors: list[dict[str, str]] = []
        try:
            if not self.service.claim_run(run_id):
                return
            run = self.service.get_run(run_id)
            errors = list((run.get("result") or {}).get("errors", []))
            image_ids = run["params"]["image_ids"]
            params = run["params"]["params"]
            completed = max(0, min(int(run["completed"]), len(image_ids)))
            for index, image_id in enumerate(image_ids[completed:], completed + 1):
                current = self.service.get_run(run_id)
                if current["cancel_requested"]:
                    self.service.update_run(
                        run_id,
                        status="cancelled",
                        result_json=json.dumps({"errors": errors}, ensure_ascii=False),
                    )
                    return
                try:
                    self.service.auto_label_image(
                        run["project_id"],
                        image_id,
                        run["model_id"],
                        params,
                        run_id=run_id,
                        model_revision_id=run["model_revision_id"],
                    )
                except Exception as exc:  # keep a large batch moving while retaining provenance
                    errors.append({"image_id": image_id, "error": str(exc)})
                self.service.update_run(
                    run_id,
                    completed=index,
                    result_json=json.dumps({"errors": errors}, ensure_ascii=False),
                )
            self.service.update_run(
                run_id,
                status="completed_with_errors" if errors else "completed",
                result_json=json.dumps({"errors": errors}, ensure_ascii=False),
            )
        except Exception as exc:
            self.service.update_run(run_id, status="failed", error=str(exc))
        finally:
            with self._lock:
                self._submitted.discard(run_id)

    def shutdown(self) -> None:
        with self._lock:
            active = list(self._submitted)
        for run_id in active:
            try:
                self.service.cancel_run(run_id)
            except KeyError:
                pass
        self.executor.shutdown(wait=True, cancel_futures=False)


def create_app(service: AnnotationService, static_dir: Path | None = None) -> FastAPI:
    jobs = JobManager(service)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        for run_id in service.recover_interrupted_runs():
            jobs.submit(run_id)
        try:
            yield
        finally:
            jobs.shutdown()

    app = FastAPI(
        title="RKHunter Annotation Tool",
        version=TOOL_VERSION,
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    static_dir = static_dir or Path(__file__).with_name("static")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {
            "ready": True,
            "tool_version": TOOL_VERSION,
            "api_version": API_VERSION,
            "database_schema_version": service.database.schema_version(),
            "supported_database_schema_version": DATABASE_SCHEMA_VERSION,
            "adapters": default_registry.names(),
        }

    @app.get("/api/v1/projects")
    def list_projects() -> list[dict[str, Any]]:
        return service.list_projects()

    @app.post("/api/v1/projects")
    def create_project(payload: ProjectCreate) -> dict[str, Any]:
        try:
            return service.register_project(
                payload.id,
                payload.name,
                payload.dataset_root,
                payload.classes,
                image_dir=payload.image_dir,
                label_dir=payload.label_dir,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/projects/{project_id}/import-yolo")
    def import_yolo(project_id: str, payload: ImportRequest) -> dict[str, int]:
        try:
            return service.import_yolo(project_id, imported_status=payload.imported_status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/projects/{project_id}/images")
    def list_images(
        project_id: str,
        status: str | None = None,
        split: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        return service.list_images(
            project_id, status=status, split=split, offset=offset, limit=limit
        )

    @app.get("/api/v1/projects/{project_id}/images/{image_id}")
    def get_image(project_id: str, image_id: str) -> dict[str, Any]:
        try:
            return service.get_image(project_id, image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/projects/{project_id}/images/{image_id}/content")
    def image_content(project_id: str, image_id: str) -> FileResponse:
        try:
            path = service.image_path(project_id, image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path)

    @app.put("/api/v1/projects/{project_id}/images/{image_id}/annotations")
    def save_annotations(
        project_id: str, image_id: str, payload: AnnotationSave
    ) -> dict[str, Any]:
        try:
            return service.save_annotations(
                project_id,
                image_id,
                [value.model_dump() for value in payload.annotations],
                expected_revision=payload.expected_revision,
                status=payload.status,
                actor=payload.actor,
            )
        except RevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/projects/{project_id}/images/{image_id}/auto-label")
    def auto_label(
        project_id: str, image_id: str, payload: AutoLabelRequest
    ) -> dict[str, Any]:
        try:
            return service.auto_label_image(
                project_id,
                image_id,
                payload.model_id,
                payload.params,
                replace_auto=payload.replace_auto,
            )
        except RevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/projects/{project_id}/stats")
    def stats(project_id: str) -> dict[str, Any]:
        return service.stats(project_id)

    @app.get("/api/v1/projects/{project_id}/events")
    def events(project_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return service.audit_events(project_id, limit)

    @app.get("/api/v1/models")
    def list_models() -> list[dict[str, Any]]:
        return service.list_models()

    @app.post("/api/v1/models")
    def register_model(payload: ModelCreate) -> dict[str, Any]:
        try:
            return service.register_model(
                payload.id,
                payload.name,
                payload.adapter,
                payload.weights_path,
                payload.config,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/models/{model_id}/revisions")
    def list_model_revisions(model_id: str) -> list[dict[str, Any]]:
        try:
            return service.list_model_revisions(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/autolabel-runs")
    def create_run(payload: RunCreate) -> dict[str, Any]:
        try:
            run_id = service.create_run(
                payload.project_id, payload.model_id, payload.image_ids, payload.params
            )
            jobs.submit(run_id)
            return service.get_run(run_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/autolabel-runs")
    def list_runs(project_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        try:
            return service.list_runs(project_id, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/autolabel-runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return service.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/autolabel-runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        try:
            return service.cancel_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/projects/{project_id}/exports/yolo")
    def export_yolo(project_id: str, payload: ExportRequest) -> dict[str, Any]:
        try:
            return service.export_yolo(
                project_id,
                payload.output_root,
                reviewed_only=payload.reviewed_only,
                copy_mode=payload.copy_mode,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app
