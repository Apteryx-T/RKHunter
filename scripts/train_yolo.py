from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", "CLOCK$"}


def repo_path(value: str | Path, *, must_be_file: bool = False) -> Path:
    path = Path(value)
    resolved = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(REPO)
    except ValueError as exc:
        raise SystemExit(f"Path must stay inside the repository: {resolved}") from exc
    if must_be_file and not resolved.is_file():
        raise SystemExit(f"Required local file not found: {resolved}")
    return resolved


def safe_run_name(value: str) -> str:
    windows_stem = value.split(".", 1)[0].upper()
    reserved = windows_stem in WINDOWS_RESERVED_NAMES or bool(
        re.fullmatch(r"(?:COM|LPT)[1-9]", windows_stem)
    )
    if (
        not RUN_NAME_PATTERN.fullmatch(value)
        or value in {".", ".."}
        or value.endswith(".")
        or reserved
    ):
        raise SystemExit(
            "Run name must be a portable directory name containing only letters, "
            "numbers, dot, underscore, or dash"
        )
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_file(root: Path, relative_value: str, field_name: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute():
        raise SystemExit(f"Manifest {field_name} must be relative: {relative_value}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"Manifest {field_name} escapes its revision: {relative_value}") from exc
    if not resolved.is_file():
        raise SystemExit(f"Manifest file is missing: {resolved}")
    return resolved


def verify_export_manifest(data_path: Path, manifest: dict) -> None:
    root = data_path.parent.resolve()
    if int(manifest.get("annotation_schema_version", 0)) < 2:
        raise SystemExit("Annotation export predates canonical hashes; create a new export")
    if manifest.get("hash_algorithm") != "sha256":
        raise SystemExit("Annotation export does not declare SHA256 canonical hashes")
    if file_sha256(data_path) != manifest.get("dataset_yaml_sha256"):
        raise SystemExit("Annotation export dataset.yaml hash mismatch")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit("Annotation export manifest has no items")
    for item in items:
        image_path = manifest_file(root, item.get("output", ""), "output")
        label_path = manifest_file(root, item.get("label_output", ""), "label_output")
        if file_sha256(image_path) != item.get("output_sha256"):
            raise SystemExit(f"Annotation export image hash mismatch: {image_path}")
        if file_sha256(label_path) != item.get("label_sha256"):
            raise SystemExit(f"Annotation export label hash mismatch: {label_path}")


def prepare_training_copy(data_path: Path, manifest: dict, project_path: Path) -> Path:
    source_root = data_path.parent.resolve()
    safe_revision = re.sub(r"[^A-Za-z0-9._-]+", "-", str(manifest["revision"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work_root = project_path / "_dataset_work" / f"{safe_revision}-{stamp}"
    for split in ("train", "val", "test"):
        (work_root / "images" / split).mkdir(parents=True, exist_ok=False)
        (work_root / "labels" / split).mkdir(parents=True, exist_ok=False)
    for item in manifest["items"]:
        for field_name in ("output", "label_output"):
            source = manifest_file(source_root, item[field_name], field_name)
            destination = work_root / item[field_name]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    shutil.copy2(data_path, work_root / "dataset.yaml")
    shutil.copy2(data_path.with_name("manifest.json"), work_root / "manifest.json")
    return work_root / "dataset.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for RKHunter.")
    parser.add_argument("--data", default="configs/dataset.yaml", help="Path to YOLO dataset YAML.")
    parser.add_argument(
        "--model", default="models/yolov8n.pt", help="Explicit local base model path."
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument("--device", default="cpu", help="Training device, such as cpu or 0.")
    parser.add_argument("--project", default="experiments/yolo", help="Output project directory.")
    parser.add_argument("--name", default="first-detector", help="Run name.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify and stage a versioned export without starting training.",
    )
    args = parser.parse_args()

    cache_root = repo_path("models/annotation-tool-cache")
    yolo_cache = repo_path(cache_root / "ultralytics")
    mpl_cache = repo_path(cache_root / "matplotlib")
    torch_cache = repo_path(cache_root / "torch")
    yolo_cache.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_cache)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"

    data_path = repo_path(args.data, must_be_file=True)
    project_path = repo_path(args.project)
    run_name = safe_run_name(args.name)
    repo_path(project_path / run_name)
    manifest_path = data_path.with_name("manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("train_ready", False):
            issues = "; ".join(manifest.get("readiness_issues", [])) or "unknown readiness failure"
            raise SystemExit(f"Annotation export is not train-ready: {issues}")
        verify_export_manifest(data_path, manifest)
        data_path = prepare_training_copy(data_path, manifest, project_path)
        print(f"Verified training dataset copy: {data_path}")

    if args.verify_only:
        print(f"Dataset verification complete: {data_path}")
        return

    model_path = repo_path(args.model, must_be_file=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: python -m pip install ultralytics"
        ) from exc

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        workers=args.workers,
        device=args.device,
        project=str(project_path),
        name=run_name,
    )


if __name__ == "__main__":
    main()
