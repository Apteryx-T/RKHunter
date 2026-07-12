from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from . import DATABASE_SCHEMA_VERSION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dataset_root TEXT NOT NULL,
    image_dir TEXT NOT NULL DEFAULT 'images',
    label_dir TEXT NOT NULL DEFAULT 'labels',
    classes_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    split TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unreviewed',
    revision INTEGER NOT NULL DEFAULT 0,
    warning TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, rel_path)
);

CREATE INDEX IF NOT EXISTS idx_images_project_status
ON images(project_id, status, rel_path);

CREATE TABLE IF NOT EXISTS model_registry (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    adapter TEXT NOT NULL,
    weights_path TEXT NOT NULL,
    version TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS autolabel_runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES model_registry(id),
    status TEXT NOT NULL,
    total INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    params_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    class_id INTEGER NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    geometry_type TEXT NOT NULL DEFAULT 'box',
    polygon_json TEXT,
    source TEXT NOT NULL,
    confidence REAL,
    status TEXT NOT NULL DEFAULT 'draft',
    model_id TEXT REFERENCES model_registry(id),
    run_id TEXT REFERENCES autolabel_runs(id),
    warning TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations(image_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    image_id TEXT,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    created_at TEXT NOT NULL
);
"""

SCHEMA_V2 = """
CREATE TABLE model_revisions (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    name TEXT NOT NULL,
    adapter TEXT NOT NULL,
    weights_path TEXT NOT NULL,
    version TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_model_revisions_model
ON model_revisions(model_id, created_at);

ALTER TABLE model_registry ADD COLUMN revision_id TEXT REFERENCES model_revisions(id);
ALTER TABLE autolabel_runs ADD COLUMN model_revision_id TEXT REFERENCES model_revisions(id);
ALTER TABLE annotations ADD COLUMN model_revision_id TEXT REFERENCES model_revisions(id);
ALTER TABLE images ADD COLUMN source_label_state TEXT;
ALTER TABLE images ADD COLUMN source_label_sha256 TEXT;

INSERT OR IGNORE INTO model_revisions(
    id, model_id, name, adapter, weights_path, version, sha256, config_json, created_at
)
SELECT
    id || '-legacy-' || substr(sha256, 1, 12), id, name, adapter, weights_path,
    version, sha256, config_json, created_at
FROM model_registry;

UPDATE model_registry
SET revision_id = id || '-legacy-' || substr(sha256, 1, 12)
WHERE revision_id IS NULL;

UPDATE annotations
SET model_revision_id = (
    SELECT revision_id FROM model_registry WHERE model_registry.id = annotations.model_id
)
WHERE model_id IS NOT NULL AND model_revision_id IS NULL;

UPDATE autolabel_runs
SET model_revision_id = (
    SELECT revision_id FROM model_registry WHERE model_registry.id = autolabel_runs.model_id
)
WHERE model_revision_id IS NULL;
"""

MIGRATIONS = {1: SCHEMA_V1, 2: SCHEMA_V2}

REQUIRED_SCHEMA = {
    "projects": {"id", "classes_json"},
    "images": {
        "id",
        "project_id",
        "revision",
        "source_label_state",
        "source_label_sha256",
    },
    "model_registry": {"id", "revision_id"},
    "model_revisions": {"id", "model_id", "sha256", "config_json"},
    "autolabel_runs": {"id", "model_revision_id"},
    "annotations": {"id", "image_id", "model_revision_id"},
    "audit_log": {"id", "project_id"},
}


def migration_statements(script: str) -> Iterator[str]:
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                yield statement
            buffer = ""
    if buffer.strip():
        raise RuntimeError("Incomplete SQL migration statement")


class AnnotationDatabase:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        # Refuse a future schema before executing DDL or enabling WAL.
        current = 0
        if self.path.exists() and self.path.stat().st_size:
            read_only = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
            try:
                has_meta = read_only.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'"
                ).fetchone()
                if has_meta:
                    row = read_only.execute(
                        "SELECT value FROM schema_meta WHERE key='schema_version'"
                    ).fetchone()
                    current = int(row[0]) if row else 0
                    if current > DATABASE_SCHEMA_VERSION:
                        raise RuntimeError(
                            f"Database schema {current} is newer than supported "
                            f"{DATABASE_SCHEMA_VERSION}"
                        )
            finally:
                read_only.close()
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("BEGIN IMMEDIATE")
            has_meta = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'"
            ).fetchone()
            locked_current = 0
            if has_meta:
                row = connection.execute(
                    "SELECT value FROM schema_meta WHERE key='schema_version'"
                ).fetchone()
                locked_current = int(row[0]) if row else 0
            if locked_current > DATABASE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema {locked_current} is newer than supported "
                    f"{DATABASE_SCHEMA_VERSION}"
                )
            for version in range(locked_current + 1, DATABASE_SCHEMA_VERSION + 1):
                migration = MIGRATIONS.get(version)
                if migration is None:
                    raise RuntimeError(
                        f"No database migration is available for version {version}"
                    )
                for statement in migration_statements(migration):
                    connection.execute(statement)
                connection.execute(
                    "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                    (str(version),),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        self._validate_schema()

    def _validate_schema(self) -> None:
        connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        try:
            for table, expected_columns in REQUIRED_SCHEMA.items():
                columns = {
                    row[1]
                    for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
                }
                missing = expected_columns - columns
                if missing:
                    raise RuntimeError(
                        f"Database schema {DATABASE_SCHEMA_VERSION} is missing "
                        f"{table} columns: {sorted(missing)}"
                    )
        finally:
            connection.close()

    def schema_version(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
        return int(row["value"])

    @staticmethod
    def json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
