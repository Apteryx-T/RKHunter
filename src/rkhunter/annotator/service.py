from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from . import ANNOTATION_SCHEMA_VERSION, TOOL_VERSION
from .adapters import AdapterContext, default_registry
from .database import AnnotationDatabase, utc_now
from .models import BoxProposal, ModelDescriptor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
MODEL_ID_PATTERN = PROJECT_ID_PATTERN


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_version(path: Path) -> tuple[str, str]:
    if path.is_file():
        sha = file_sha256(path)
    elif path.is_dir():
        digest = hashlib.sha256()
        files = sorted(value for value in path.rglob("*") if value.is_file())
        if not files:
            raise ValueError(f"model directory is empty: {path}")
        for file_path in files:
            resolved = file_path.resolve()
            try:
                relative = resolved.relative_to(path)
            except ValueError as exc:
                raise ValueError(f"model file escapes registered directory: {file_path}") from exc
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(resolved.stat().st_size).encode("ascii"))
            digest.update(b"\0")
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        sha = digest.hexdigest()
    else:
        raise ValueError(f"model path must be a file or directory: {path}")
    return sha[:12], sha


def directory_stat_signature(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(value for value in path.rglob("*") if value.is_file()):
        resolved = file_path.resolve()
        try:
            relative = resolved.relative_to(path)
        except ValueError as exc:
            raise ValueError(f"model file escapes registered directory: {file_path}") from exc
        stat = resolved.stat()
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode("ascii"))
    return digest.hexdigest()


class RevisionConflict(RuntimeError):
    pass


class AnnotationService:
    def __init__(self, workspace_root: Path, database_path: Path):
        self.workspace_root = workspace_root.resolve()
        self.database = AnnotationDatabase(self.safe_path(database_path))
        self._write_lock = threading.RLock()
        self._adapter_cache: dict[tuple[str, str], Any] = {}
        self._model_directory_signatures: dict[str, str] = {}
        default_registry.load_entry_points()

    def safe_path(self, value: str | Path, *, must_exist: bool = False) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.workspace_root / path
        path = path.resolve()
        try:
            path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"path must stay inside workspace: {path}") from exc
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path

    def register_project(
        self,
        project_id: str,
        name: str,
        dataset_root: str | Path,
        classes: dict[int, str],
        *,
        image_dir: str = "images",
        label_dir: str = "labels",
    ) -> dict[str, Any]:
        if not PROJECT_ID_PATTERN.fullmatch(project_id):
            raise ValueError("project id must match [a-z0-9][a-z0-9_-]{1,63}")
        root = self.safe_path(dataset_root, must_exist=True)
        image_dir, image_root = self._dataset_subdir(root, image_dir, "image_dir")
        label_dir, _ = self._dataset_subdir(root, label_dir, "label_dir")
        if not image_root.is_dir():
            raise FileNotFoundError(f"image directory not found: {image_root}")
        normalized_classes = {int(key): str(value).strip() for key, value in classes.items()}
        if not normalized_classes or any(
            key < 0 or not value for key, value in normalized_classes.items()
        ):
            raise ValueError("classes must be a non-empty non-negative id-to-name mapping")
        if sorted(normalized_classes) != list(range(len(normalized_classes))):
            raise ValueError("YOLO class ids must be contiguous and start at 0")
        now = utc_now()
        classes_json = self.database.json(
            {str(key): normalized_classes[key] for key in sorted(normalized_classes)}
        )
        with self._write_lock, self.database.connection() as connection:
            existing = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            if existing and Path(existing["dataset_root"]).resolve() != root:
                raise ValueError("project id already points to another dataset root")
            if existing:
                image_count = connection.execute(
                    "SELECT COUNT(*) FROM images WHERE project_id = ?", (project_id,)
                ).fetchone()[0]
                schema_changed = (
                    existing["image_dir"] != image_dir
                    or existing["label_dir"] != label_dir
                    or existing["classes_json"] != classes_json
                )
                if image_count and schema_changed:
                    raise ValueError(
                        "project dataset directories and class taxonomy are immutable after import"
                    )
            connection.execute(
                """
                INSERT INTO projects(id, name, dataset_root, image_dir, label_dir, classes_json, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, image_dir=excluded.image_dir, label_dir=excluded.label_dir,
                    classes_json=excluded.classes_json, updated_at=excluded.updated_at
                """,
                (
                    project_id,
                    name,
                    str(root),
                    image_dir,
                    label_dir,
                    classes_json,
                    now,
                    now,
                ),
            )
            self._audit(connection, project_id, None, "project_registered", None, {"root": str(root)})
        return self.get_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute("SELECT * FROM projects ORDER BY name").fetchall()
        return [self._project_row(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self.database.connection() as connection:
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            raise KeyError(f"project not found: {project_id}")
        return self._project_row(row)

    def import_yolo(self, project_id: str, *, imported_status: str = "auto_labeled") -> dict[str, int]:
        if imported_status not in {"unreviewed", "auto_labeled", "reviewed"}:
            raise ValueError("invalid imported status")
        project = self.get_project(project_id)
        dataset_root = self.safe_path(project["dataset_root"], must_exist=True)
        _, image_root = self._dataset_subdir(
            dataset_root, project["image_dir"], "stored image_dir"
        )
        _, label_root = self._dataset_subdir(
            dataset_root, project["label_dir"], "stored label_dir"
        )
        valid_classes = {int(key) for key in project["classes"]}
        background_ids = {
            int(key)
            for key, value in project["classes"].items()
            if str(value).lower() == "background"
        }
        counts = {
            "images": 0,
            "labels": 0,
            "boxes": 0,
            "skipped_existing": 0,
            "metadata_backfilled": 0,
        }
        image_paths = sorted(
            path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        with self._write_lock, self.database.connection() as connection:
            for discovered_image_path in image_paths:
                rel_path = discovered_image_path.relative_to(image_root).as_posix()
                image_path = discovered_image_path.resolve()
                try:
                    image_path.relative_to(image_root)
                except ValueError as exc:
                    raise ValueError(
                        f"image symlink escapes project image root: {discovered_image_path}"
                    ) from exc
                split = rel_path.split("/", 1)[0] if "/" in rel_path else "unsplit"
                sha = file_sha256(image_path)
                image_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{project_id}:{rel_path}:{sha}").hex
                with Image.open(image_path) as image:
                    width, height = image.size
                now = utc_now()
                discovered_label_path = label_root / Path(rel_path).with_suffix(".txt")
                label_path = discovered_label_path.resolve()
                try:
                    label_path.relative_to(label_root)
                except ValueError as exc:
                    raise ValueError(
                        f"label symlink escapes project label root: {discovered_label_path}"
                    ) from exc
                if label_path.is_file():
                    label_sha = file_sha256(label_path)
                    label_state = (
                        "labeled"
                        if label_path.read_text(encoding="utf-8-sig").strip()
                        else "empty"
                    )
                else:
                    label_sha = None
                    label_state = "missing"
                existing = connection.execute(
                    "SELECT * FROM images WHERE project_id = ? AND rel_path = ?", (project_id, rel_path)
                ).fetchone()
                if existing:
                    if (
                        existing["sha256"] != sha
                        or existing["width"] != width
                        or existing["height"] != height
                    ):
                        raise ValueError(
                            f"source image changed after import: {rel_path}; use a new path or explicit migration"
                        )
                    if existing["source_label_state"] is None:
                        connection.execute(
                            """
                            UPDATE images SET source_label_state=?, source_label_sha256=? WHERE id=?
                            """,
                            (label_state, label_sha, existing["id"]),
                        )
                        counts["metadata_backfilled"] += 1
                    elif (
                        existing["source_label_state"] != label_state
                        or existing["source_label_sha256"] != label_sha
                    ):
                        raise ValueError(
                            f"source label changed after import: {rel_path}; use an explicit re-import workflow"
                        )
                    counts["skipped_existing"] += 1
                    continue
                connection.execute(
                    """
                    INSERT INTO images(
                        id, project_id, rel_path, split, width, height, sha256, status,
                        revision, created_at, updated_at, source_label_state, source_label_sha256
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                    """,
                    (
                        image_id,
                        project_id,
                        rel_path,
                        split,
                        width,
                        height,
                        sha,
                        "unreviewed" if label_state == "missing" else imported_status,
                        now,
                        now,
                        label_state,
                        label_sha,
                    ),
                )
                counts["images"] += 1
                if not label_path.is_file():
                    continue
                counts["labels"] += 1
                for proposal in self._parse_yolo_boxes(label_path, width, height):
                    if proposal.class_id not in valid_classes:
                        raise ValueError(
                            f"unknown class {proposal.class_id} in {label_path}"
                        )
                    if proposal.class_id in background_ids:
                        raise ValueError(
                            f"background must use an empty label file, not a box: {label_path}"
                        )
                    self._insert_annotation(
                        connection,
                        image_id,
                        proposal,
                        status="approved" if imported_status == "reviewed" else "draft",
                        run_id=None,
                    )
                    counts["boxes"] += 1
            self._audit(connection, project_id, None, "yolo_import", None, counts)
        return counts

    def list_images(
        self,
        project_id: str,
        *,
        status: str | None = None,
        split: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        clauses = ["i.project_id = ?"]
        params: list[Any] = [project_id]
        if status:
            clauses.append("i.status = ?")
            params.append(status)
        if split:
            clauses.append("i.split = ?")
            params.append(split)
        where = " AND ".join(clauses)
        with self.database.connection() as connection:
            total = connection.execute(f"SELECT COUNT(*) count FROM images i WHERE {where}", params).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT i.*, COUNT(a.id) annotation_count,
                       SUM(CASE WHEN a.warning IS NOT NULL THEN 1 ELSE 0 END) warning_count
                FROM images i LEFT JOIN annotations a ON a.image_id = i.id
                WHERE {where}
                GROUP BY i.id ORDER BY i.rel_path LIMIT ? OFFSET ?
                """,
                (*params, max(1, min(limit, 500)), max(0, offset)),
            ).fetchall()
        return {"total": total, "items": [dict(row) for row in rows]}

    def get_image(self, project_id: str, image_id: str) -> dict[str, Any]:
        with self.database.connection() as connection:
            image_row = connection.execute(
                "SELECT * FROM images WHERE id = ? AND project_id = ?", (image_id, project_id)
            ).fetchone()
            if not image_row:
                raise KeyError("image not found")
            annotations = connection.execute(
                "SELECT * FROM annotations WHERE image_id = ? ORDER BY id", (image_id,)
            ).fetchall()
        payload = dict(image_row)
        payload["annotations"] = [dict(row) for row in annotations]
        return payload

    def image_path(self, project_id: str, image_id: str) -> Path:
        project = self.get_project(project_id)
        image = self.get_image(project_id, image_id)
        dataset_root = self.safe_path(project["dataset_root"], must_exist=True)
        _, root = self._dataset_subdir(
            dataset_root, project["image_dir"], "stored image_dir"
        )
        path = (root / image["rel_path"]).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("stored image path escapes project image root") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def save_annotations(
        self,
        project_id: str,
        image_id: str,
        annotations: Iterable[dict[str, Any]],
        *,
        expected_revision: int,
        status: str = "reviewed",
        actor: str = "local-user",
    ) -> dict[str, Any]:
        if status not in {"unreviewed", "auto_labeled", "reviewed", "rejected"}:
            raise ValueError("invalid image status")
        with self._write_lock, self.database.connection() as connection:
            image = connection.execute(
                "SELECT * FROM images WHERE id = ? AND project_id = ?", (image_id, project_id)
            ).fetchone()
            if not image:
                raise KeyError("image not found")
            if image["revision"] != expected_revision:
                raise RevisionConflict(
                    f"revision changed: expected {expected_revision}, current {image['revision']}"
                )
            before_rows = connection.execute(
                "SELECT * FROM annotations WHERE image_id = ? ORDER BY id", (image_id,)
            ).fetchall()
            before = [dict(row) for row in before_rows]
            proposals: list[BoxProposal] = []
            project_classes = self.get_project(project_id)["classes"]
            valid_classes = {int(key) for key in project_classes}
            background_ids = {
                int(key)
                for key, value in project_classes.items()
                if str(value).lower() == "background"
            }
            for value in annotations:
                proposal = BoxProposal(
                    class_id=int(value["class_id"]),
                    x1=float(value["x1"]),
                    y1=float(value["y1"]),
                    x2=float(value["x2"]),
                    y2=float(value["y2"]),
                    confidence=float(value["confidence"]) if value.get("confidence") is not None else None,
                    source=str(value.get("source") or "manual"),
                    model_id=value.get("model_id"),
                    model_revision_id=value.get("model_revision_id"),
                    warning=value.get("warning"),
                )
                proposal.validate(image["width"], image["height"])
                if proposal.class_id not in valid_classes:
                    raise ValueError(f"unknown class id: {proposal.class_id}")
                if proposal.class_id in background_ids:
                    raise ValueError(
                        "background must be represented by an empty annotation list"
                    )
                if bool(proposal.model_id) != bool(proposal.model_revision_id):
                    raise ValueError(
                        "model_id and model_revision_id must be provided together"
                    )
                if proposal.model_revision_id:
                    revision_row = connection.execute(
                        "SELECT model_id FROM model_revisions WHERE id=?",
                        (proposal.model_revision_id,),
                    ).fetchone()
                    if not revision_row or revision_row["model_id"] != proposal.model_id:
                        raise ValueError(
                            "model_revision_id does not belong to the supplied model_id"
                        )
                proposals.append(proposal)
            if status == "rejected" and proposals:
                raise ValueError("rejected images must have an empty annotation list")
            new_revision = expected_revision + 1
            claimed = connection.execute(
                """
                UPDATE images SET status=?, revision=?, warning=NULL, updated_at=?
                WHERE id=? AND project_id=? AND revision=?
                """,
                (status, new_revision, utc_now(), image_id, project_id, expected_revision),
            )
            if claimed.rowcount != 1:
                raise RevisionConflict(
                    f"revision changed while saving image {image_id}; reload and retry"
                )
            connection.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
            annotation_status = "approved" if status == "reviewed" else "draft"
            for proposal in proposals:
                self._insert_annotation(
                    connection, image_id, proposal, status=annotation_status, run_id=None
                )
            after = [proposal.to_dict() | {"status": annotation_status} for proposal in proposals]
            self._audit(connection, project_id, image_id, "annotations_saved", before, after, actor)
        return self.get_image(project_id, image_id)

    def register_model(
        self,
        model_id: str,
        name: str,
        adapter: str,
        weights_path: str | Path,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not MODEL_ID_PATTERN.fullmatch(model_id):
            raise ValueError("model id must match [a-z0-9][a-z0-9_-]{1,63}")
        if adapter not in default_registry.names():
            raise ValueError(f"unsupported adapter {adapter}; available: {default_registry.names()}")
        weights = self.safe_path(weights_path, must_exist=True)
        if adapter == "ultralytics_yolo" and not weights.is_file():
            raise ValueError("YOLO weights_path must be a local file")
        version, sha = model_version(weights)
        revision_config = dict(config or {})
        revision_config["_rkhunter_tool_version"] = TOOL_VERSION
        revision_config["_adapter_version"] = default_registry.version(adapter)
        config_json = json.dumps(
            revision_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        revision_digest = hashlib.sha256(
            json.dumps(
                {
                    "model_id": model_id,
                    "adapter": adapter,
                    "weights_path": str(weights),
                    "weights_sha256": sha,
                    "config": json.loads(config_json),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        revision_id = f"{model_id}-{revision_digest[:16]}"
        now = utc_now()
        with self._write_lock, self.database.connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO model_revisions(
                    id, model_id, name, adapter, weights_path, version, sha256,
                    config_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision_id,
                    model_id,
                    name,
                    adapter,
                    str(weights),
                    version,
                    sha,
                    config_json,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO model_registry(
                    id, name, adapter, weights_path, version, sha256, config_json,
                    active, created_at, updated_at, revision_id
                ) VALUES(?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name, adapter=excluded.adapter,
                    weights_path=excluded.weights_path, version=excluded.version,
                    sha256=excluded.sha256, config_json=excluded.config_json,
                    active=1, updated_at=excluded.updated_at,
                    revision_id=excluded.revision_id
                """,
                (
                    model_id,
                    name,
                    adapter,
                    str(weights),
                    version,
                    sha,
                    config_json,
                    now,
                    now,
                    revision_id,
                ),
            )
            self._audit(
                connection,
                "system",
                None,
                "model_registered",
                None,
                {"id": model_id, "revision_id": revision_id, "sha": sha},
            )
        self._adapter_cache = {key: value for key, value in self._adapter_cache.items() if key[0] != model_id}
        return self.get_model(model_id)

    def list_models(self) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute("SELECT * FROM model_registry ORDER BY name").fetchall()
        return [self._model_row(row) for row in rows]

    def get_model(self, model_id: str) -> dict[str, Any]:
        with self.database.connection() as connection:
            row = connection.execute("SELECT * FROM model_registry WHERE id = ?", (model_id,)).fetchone()
        if not row:
            raise KeyError("model not found")
        return self._model_row(row)

    def list_model_revisions(self, model_id: str) -> list[dict[str, Any]]:
        self.get_model(model_id)
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM model_revisions WHERE model_id=? ORDER BY created_at DESC",
                (model_id,),
            ).fetchall()
        return [self._model_revision_row(row) for row in rows]

    def get_model_revision(self, revision_id: str) -> dict[str, Any]:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM model_revisions WHERE id=?", (revision_id,)
            ).fetchone()
        if not row:
            raise KeyError("model revision not found")
        return self._model_revision_row(row)

    def auto_label_image(
        self,
        project_id: str,
        image_id: str,
        model_id: str,
        params: dict[str, Any],
        *,
        run_id: str | None = None,
        model_revision_id: str | None = None,
        replace_auto: bool = True,
    ) -> dict[str, Any]:
        image = self.get_image(project_id, image_id)
        if image["status"] in {"reviewed", "rejected"}:
            raise ValueError(
                "human-reviewed image is locked; change its status explicitly before auto-labeling"
            )
        model_alias = self.get_model(model_id)
        revision_id = model_revision_id or model_alias["revision_id"]
        model_data = self.get_model_revision(revision_id)
        if model_data["model_id"] != model_id:
            raise ValueError("model revision does not belong to the requested model alias")
        descriptor = ModelDescriptor(
            id=model_id,
            revision_id=model_data["id"],
            name=model_data["name"],
            adapter=model_data["adapter"],
            weights_path=model_data["weights_path"],
            version=model_data["version"],
            sha256=model_data["sha256"],
            config=model_data["config"],
            active=model_alias["active"],
        )
        cache_key = (descriptor.id, descriptor.revision_id)
        self._verify_registered_model_content(descriptor)
        adapter = self._adapter_cache.get(cache_key)
        if adapter is None:
            adapter = default_registry.create(
                descriptor.adapter, AdapterContext(model=descriptor, device=str(params.get("device", "cpu")))
            )
            self._adapter_cache[cache_key] = adapter
        proposals = adapter.propose(
            self.image_path(project_id, image_id),
            image_width=image["width"],
            image_height=image["height"],
            params=params,
        )
        for proposal in proposals:
            proposal.model_id = descriptor.id
            proposal.model_revision_id = descriptor.revision_id
            proposal.validate(image["width"], image["height"])
        project = self.get_project(project_id)
        valid_classes = {int(key) for key in project["classes"]}
        background_ids = {
            int(key)
            for key, value in project["classes"].items()
            if str(value).lower() == "background"
        }
        proposals = [
            proposal
            for proposal in proposals
            if proposal.class_id in valid_classes and proposal.class_id not in background_ids
        ]
        with self._write_lock, self.database.connection() as connection:
            current = connection.execute(
                "SELECT * FROM images WHERE id = ? AND project_id = ?", (image_id, project_id)
            ).fetchone()
            if not current:
                raise KeyError("image not found")
            if current["revision"] != image["revision"]:
                raise RevisionConflict(
                    f"image {image_id} changed while the model was running; reload and retry"
                )
            approved_count = connection.execute(
                "SELECT COUNT(*) FROM annotations WHERE image_id = ? AND status = 'approved'", (image_id,)
            ).fetchone()[0]
            if current["status"] in {"reviewed", "rejected"}:
                raise ValueError(
                    "human-reviewed image is locked; change its status explicitly before auto-labeling"
                )
            if approved_count:
                raise ValueError("approved annotations exist; auto-label will not overwrite human work")
            before = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM annotations WHERE image_id = ? ORDER BY id", (image_id,)
                ).fetchall()
            ]
            claimed = connection.execute(
                """
                UPDATE images SET status='auto_labeled', revision=?, warning=?, updated_at=?
                WHERE id=? AND project_id=? AND revision=?
                  AND status NOT IN ('reviewed', 'rejected')
                """,
                (
                    image["revision"] + 1,
                    ";".join(sorted({item.warning for item in proposals if item.warning})) or None,
                    utc_now(),
                    image_id,
                    project_id,
                    image["revision"],
                ),
            )
            if claimed.rowcount != 1:
                raise RevisionConflict(
                    f"image {image_id} changed while the model was running; reload and retry"
                )
            if replace_auto:
                connection.execute(
                    "DELETE FROM annotations WHERE image_id = ? AND status = 'draft'", (image_id,)
                )
            for proposal in proposals:
                self._insert_annotation(
                    connection, image_id, proposal, status="draft", run_id=run_id
                )
            self._audit(
                connection,
                project_id,
                image_id,
                "auto_labeled",
                before,
                [proposal.to_dict() for proposal in proposals],
                actor=f"model:{model_id}@{descriptor.revision_id}",
            )
        return self.get_image(project_id, image_id)

    def _verify_registered_model_content(self, descriptor: ModelDescriptor) -> None:
        path = Path(descriptor.weights_path)
        if path.is_file():
            _, current_sha = model_version(path)
        elif path.is_dir():
            signature = directory_stat_signature(path)
            if self._model_directory_signatures.get(descriptor.revision_id) == signature:
                return
            _, current_sha = model_version(path)
            stable_signature = directory_stat_signature(path)
            if stable_signature != signature:
                raise RuntimeError("registered model directory changed while it was being verified")
        else:
            raise FileNotFoundError(path)
        if current_sha != descriptor.sha256:
            raise RuntimeError(
                "registered model content changed; register a new immutable revision before inference"
            )
        if path.is_dir():
            self._model_directory_signatures[descriptor.revision_id] = stable_signature

    def create_run(
        self, project_id: str, model_id: str, image_ids: list[str], params: dict[str, Any]
    ) -> str:
        self.get_project(project_id)
        model = self.get_model(model_id)
        if not image_ids:
            raise ValueError("an auto-label run requires at least one image")
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("auto-label run image ids must be unique")
        run_id = uuid.uuid4().hex
        now = utc_now()
        payload = {"image_ids": image_ids, "params": params}
        with self._write_lock, self.database.connection() as connection:
            found_ids: set[str] = set()
            for offset in range(0, len(image_ids), 400):
                chunk = image_ids[offset : offset + 400]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT id FROM images WHERE project_id=? AND id IN ({placeholders})",
                    (project_id, *chunk),
                ).fetchall()
                found_ids.update(row["id"] for row in rows)
            if len(found_ids) != len(image_ids):
                raise ValueError("all auto-label run images must belong to the project")
            connection.execute(
                """
                INSERT INTO autolabel_runs(
                    id, project_id, model_id, status, total, completed, params_json,
                    created_at, updated_at, model_revision_id
                ) VALUES(?, ?, ?, 'queued', ?, 0, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    project_id,
                    model_id,
                    len(image_ids),
                    self.database.json(payload),
                    now,
                    now,
                    model["revision_id"],
                ),
            )
        return run_id

    def claim_run(self, run_id: str) -> bool:
        with self._write_lock, self.database.connection() as connection:
            claimed = connection.execute(
                """
                UPDATE autolabel_runs SET status='running', updated_at=?
                WHERE id=? AND status='queued'
                """,
                (utc_now(), run_id),
            )
        return claimed.rowcount == 1

    def recover_interrupted_runs(self) -> list[str]:
        with self._write_lock, self.database.connection() as connection:
            interrupted = connection.execute(
                "SELECT id, project_id FROM autolabel_runs WHERE status='running'"
            ).fetchall()
            connection.execute(
                """
                UPDATE autolabel_runs
                SET status='queued', error='interrupted process; queued for recovery', updated_at=?
                WHERE status='running'
                """,
                (utc_now(),),
            )
            for row in interrupted:
                self._audit(
                    connection,
                    row["project_id"],
                    None,
                    "autolabel_run_recovered",
                    {"run_id": row["id"], "status": "running"},
                    {"run_id": row["id"], "status": "queued"},
                )
            queued = connection.execute(
                "SELECT id FROM autolabel_runs WHERE status='queued' ORDER BY created_at"
            ).fetchall()
        return [row["id"] for row in queued]

    def list_runs(
        self, project_id: str | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        query = "SELECT id FROM autolabel_runs"
        params: list[Any] = []
        if project_id:
            self.get_project(project_id)
            query += " WHERE project_id=?"
            params.append(project_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(limit, 1000)))
        with self.database.connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self.get_run(row["id"]) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.database.connection() as connection:
            row = connection.execute("SELECT * FROM autolabel_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            raise KeyError("run not found")
        value = dict(row)
        value["params"] = json.loads(value.pop("params_json"))
        value["result"] = json.loads(value.pop("result_json")) if value.get("result_json") else None
        return value

    def update_run(self, run_id: str, **values: Any) -> None:
        allowed = {"status", "completed", "result_json", "error", "cancel_requested"}
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = utc_now()
        clause = ", ".join(f"{key} = ?" for key in updates)
        with self.database.connection() as connection:
            connection.execute(
                f"UPDATE autolabel_runs SET {clause} WHERE id = ?", (*updates.values(), run_id)
            )

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        self.update_run(run_id, cancel_requested=1)
        return self.get_run(run_id)

    def export_yolo(
        self,
        project_id: str,
        output_root: str | Path,
        *,
        reviewed_only: bool = True,
        copy_mode: str = "copy",
    ) -> dict[str, Any]:
        project = self.get_project(project_id)
        if not reviewed_only:
            raise ValueError(
                "draft export is disabled because unreviewed images must not become false backgrounds"
            )
        if copy_mode != "copy":
            raise ValueError("immutable exports require copy_mode=copy")
        export_parent = self.safe_path(output_root)
        export_parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        revision = f"{project_id}-{stamp}-{uuid.uuid4().hex[:6]}"
        destination = export_parent / revision
        staging = export_parent / f".{revision}.tmp"
        with self._write_lock, self.database.connection() as connection:
            connection.execute("BEGIN")
            status_clause = "AND i.status = 'reviewed'" if reviewed_only else ""
            images = connection.execute(
                f"SELECT i.* FROM images i WHERE i.project_id = ? {status_clause} ORDER BY i.rel_path",
                (project_id,),
            ).fetchall()
            annotation_rows = connection.execute(
                """
                SELECT a.* FROM annotations a JOIN images i ON i.id = a.image_id
                WHERE i.project_id = ? AND a.status = 'approved' ORDER BY a.image_id, a.id
                """,
                (project_id,),
            ).fetchall()
        if not images:
            qualifier = "reviewed " if reviewed_only else ""
            raise ValueError(f"no {qualifier}images are available for export")
        unsupported_splits = sorted(
            {str(image["split"] or "unsplit") for image in images}
            - {"train", "val", "test"}
        )
        if unsupported_splits:
            raise ValueError(
                f"YOLO export supports only train/val/test splits: {unsupported_splits}"
            )
        by_image: dict[str, list[dict[str, Any]]] = {}
        for row in annotation_rows:
            by_image.setdefault(row["image_id"], []).append(dict(row))
        classes = {int(key): value for key, value in project["classes"].items()}
        background_ids = {key for key, value in classes.items() if value.lower() == "background"}
        detection_classes = {
            class_id: name
            for class_id, name in sorted(classes.items())
            if class_id not in background_ids
        }
        if not detection_classes:
            raise ValueError("at least one non-background detection class is required")
        class_id_map = {
            source_id: export_id
            for export_id, source_id in enumerate(detection_classes)
        }
        dataset_root = self.safe_path(project["dataset_root"], must_exist=True)
        _, source_image_root = self._dataset_subdir(
            dataset_root, project["image_dir"], "stored image_dir"
        )
        manifest: list[dict[str, Any]] = []
        box_count = 0
        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        split_box_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        output_keys: set[str] = set()
        try:
            staging.mkdir(parents=False, exist_ok=False)
            for split_name in ("train", "val", "test"):
                (staging / "images" / split_name).mkdir(parents=True)
                (staging / "labels" / split_name).mkdir(parents=True)
            for image in images:
                relative_source = Path(image["rel_path"])
                source = (source_image_root / relative_source).resolve()
                try:
                    source.relative_to(source_image_root)
                except ValueError as exc:
                    raise ValueError("stored image path escapes project image root") from exc
                if not source.is_file():
                    raise FileNotFoundError(source)
                current_sha = file_sha256(source)
                if current_sha != image["sha256"]:
                    raise ValueError(
                        f"source image changed after import: {image['rel_path']}; re-import before exporting"
                    )
                split = image["split"] or "unsplit"
                split_counts[split] = split_counts.get(split, 0) + 1
                output_relative = relative_source
                if not output_relative.parts or output_relative.parts[0] != split:
                    output_relative = Path(split) / output_relative
                image_dest = staging / "images" / output_relative
                label_dest = (
                    staging / "labels" / output_relative.with_suffix(".txt")
                )
                for output_path in (image_dest, label_dest):
                    key = output_path.relative_to(staging).as_posix().casefold()
                    if key in output_keys:
                        raise ValueError(
                            f"duplicate YOLO output path: {output_path.relative_to(staging)}"
                        )
                    output_keys.add(key)
                image_dest.parent.mkdir(parents=True, exist_ok=True)
                label_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, image_dest)
                output_image_sha = file_sha256(image_dest)
                if output_image_sha != current_sha:
                    raise RuntimeError(
                        f"copied image hash mismatch during export: {image['rel_path']}"
                    )
                lines: list[str] = []
                for annotation in by_image.get(image["id"], []):
                    if annotation["class_id"] in background_ids:
                        raise ValueError("background must use an empty label file, not a box")
                    if annotation["class_id"] not in class_id_map:
                        raise ValueError(
                            f"unknown approved class id: {annotation['class_id']}"
                        )
                    x_center = ((annotation["x1"] + annotation["x2"]) / 2) / image["width"]
                    y_center = ((annotation["y1"] + annotation["y2"]) / 2) / image["height"]
                    width = (annotation["x2"] - annotation["x1"]) / image["width"]
                    height = (annotation["y2"] - annotation["y1"]) / image["height"]
                    lines.append(
                        f"{class_id_map[annotation['class_id']]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )
                    box_count += 1
                    split_box_counts[split] = split_box_counts.get(split, 0) + 1
                label_dest.write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                output_label_sha = file_sha256(label_dest)
                manifest.append(
                    {
                        "image_id": image["id"],
                        "source": str(source.relative_to(self.workspace_root)),
                        "output": image_dest.relative_to(staging).as_posix(),
                        "label_output": label_dest.relative_to(staging).as_posix(),
                        "split": split,
                        "source_sha256": current_sha,
                        "output_sha256": output_image_sha,
                        "label_sha256": output_label_sha,
                        "boxes": len(lines),
                        "source_revision": image["revision"],
                    }
                )
            readiness_issues: list[str] = []
            if split_counts["train"] == 0:
                readiness_issues.append("missing_train_images")
            if split_box_counts["train"] == 0:
                readiness_issues.append("missing_train_boxes")
            if split_counts["val"] == 0:
                readiness_issues.append("missing_val_images")
            if split_box_counts["val"] == 0:
                readiness_issues.append("missing_val_boxes")
            train_ready = not readiness_issues
            yaml_lines = [
                f"train: {'images/train' if split_counts['train'] else 'null'}",
                f"val: {'images/val' if split_counts['val'] else 'null'}",
                f"test: {'images/test' if split_counts['test'] else 'null'}",
                "",
                "names:",
            ]
            yaml_lines.extend(
                f"  {key}: {json.dumps(value, ensure_ascii=False)}"
                for key, value in enumerate(detection_classes.values())
            )
            dataset_yaml_path = staging / "dataset.yaml"
            dataset_yaml_path.write_text(
                "\n".join(yaml_lines) + "\n", encoding="utf-8"
            )
            export_manifest = {
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                "tool_version": TOOL_VERSION,
                "project_id": project_id,
                "revision": revision,
                "created_at": utc_now(),
                "reviewed_only": reviewed_only,
                "images": len(manifest),
                "boxes": box_count,
                "split_counts": split_counts,
                "split_box_counts": split_box_counts,
                "eval_split": "val" if split_counts["val"] else None,
                "class_id_map": {str(key): value for key, value in class_id_map.items()},
                "hash_algorithm": "sha256",
                "dataset_yaml_sha256": file_sha256(dataset_yaml_path),
                "train_ready": train_ready,
                "readiness_issues": readiness_issues,
                "items": manifest,
            }
            (staging / "manifest.json").write_text(
                json.dumps(export_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            staging.replace(destination)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        audit_warning = None
        try:
            with self.database.connection() as connection:
                self._audit(
                    connection,
                    project_id,
                    None,
                    "yolo_export",
                    None,
                    {"revision": revision, "images": len(manifest), "boxes": box_count},
                )
        except Exception as exc:
            audit_warning = f"export published but audit logging failed: {exc}"
        return {
            "revision": revision,
            "path": str(destination),
            "images": len(manifest),
            "boxes": box_count,
            "split_counts": split_counts,
            "split_box_counts": split_box_counts,
            "train_ready": train_ready,
            "readiness_issues": readiness_issues,
            "audit_recorded": audit_warning is None,
            "audit_warning": audit_warning,
        }

    def stats(self, project_id: str) -> dict[str, Any]:
        with self.database.connection() as connection:
            statuses = connection.execute(
                "SELECT status, COUNT(*) count FROM images WHERE project_id=? GROUP BY status",
                (project_id,),
            ).fetchall()
            boxes = connection.execute(
                """
                SELECT a.status, COUNT(*) count FROM annotations a
                JOIN images i ON i.id=a.image_id WHERE i.project_id=? GROUP BY a.status
                """,
                (project_id,),
            ).fetchall()
        return {
            "images": {row["status"]: row["count"] for row in statuses},
            "boxes": {row["status"]: row["count"] for row in boxes},
        }

    def audit_events(self, project_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_log WHERE project_id=? ORDER BY id DESC LIMIT ?",
                (project_id, max(1, min(limit, 1000))),
            ).fetchall()
        return [dict(row) for row in rows]

    def _parse_yolo_boxes(self, label_path: Path, width: int, height: int) -> list[BoxProposal]:
        proposals: list[BoxProposal] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"unsupported YOLO label at {label_path}:{line_number}")
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(float, parts[1:])
            if not all(0 <= value <= 1 for value in (x_center, y_center, box_width, box_height)):
                raise ValueError(f"normalized coordinate out of range at {label_path}:{line_number}")
            proposal = BoxProposal(
                class_id=class_id,
                x1=(x_center - box_width / 2) * width,
                y1=(y_center - box_height / 2) * height,
                x2=(x_center + box_width / 2) * width,
                y2=(y_center + box_height / 2) * height,
                source="import",
            )
            proposal.validate(width, height)
            proposals.append(proposal)
        return proposals

    @staticmethod
    def _dataset_subdir(root: Path, value: str, field_name: str) -> tuple[str, Path]:
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError(f"{field_name} must be relative to the dataset root")
        resolved = (root / relative).resolve()
        try:
            normalized = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{field_name} must stay inside the dataset root") from exc
        return normalized.as_posix(), resolved

    @staticmethod
    def _insert_annotation(
        connection: Any,
        image_id: str,
        proposal: BoxProposal,
        *,
        status: str,
        run_id: str | None,
    ) -> None:
        now = utc_now()
        connection.execute(
            """
            INSERT INTO annotations(
                image_id, class_id, x1, y1, x2, y2, source, confidence,
                status, model_id, run_id, warning, created_at, updated_at,
                model_revision_id
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                proposal.class_id,
                proposal.x1,
                proposal.y1,
                proposal.x2,
                proposal.y2,
                proposal.source,
                proposal.confidence,
                status,
                proposal.model_id,
                run_id,
                proposal.warning,
                now,
                now,
                proposal.model_revision_id,
            ),
        )

    def _audit(
        self,
        connection: Any,
        project_id: str,
        image_id: str | None,
        action: str,
        before: Any,
        after: Any,
        actor: str = "system",
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_log(project_id, image_id, action, actor, before_json, after_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                image_id,
                action,
                actor,
                self.database.json(before) if before is not None else None,
                self.database.json(after) if after is not None else None,
                utc_now(),
            ),
        )

    @staticmethod
    def _project_row(row: Any) -> dict[str, Any]:
        value = dict(row)
        value["classes"] = {int(key): name for key, name in json.loads(value.pop("classes_json")).items()}
        return value

    @staticmethod
    def _model_row(row: Any) -> dict[str, Any]:
        value = dict(row)
        value["config"] = json.loads(value.pop("config_json"))
        value["active"] = bool(value["active"])
        return value

    @staticmethod
    def _model_revision_row(row: Any) -> dict[str, Any]:
        value = dict(row)
        value["config"] = json.loads(value.pop("config_json"))
        return value
