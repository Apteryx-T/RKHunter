from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", "CLOCK$"}
INDIRECT_SOURCE_SUFFIXES = {".streams", ".txt", ".csv"}


def repo_path(
    value: str | Path,
    *,
    must_exist: bool = False,
    must_be_file: bool = False,
) -> Path:
    path = Path(value)
    resolved = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(REPO)
    except ValueError as exc:
        raise SystemExit(f"Path must stay inside the repository: {resolved}") from exc
    if must_exist and not resolved.exists():
        raise SystemExit(f"Required local path not found: {resolved}")
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


def validate_source(path: Path) -> Path:
    if path.suffix.lower() in INDIRECT_SOURCE_SUFFIXES:
        raise SystemExit(
            f"Indirect or streaming source lists are disabled; use a local file or directory: {path}"
        )
    if path.is_dir():
        for entry in path.rglob("*"):
            resolved = entry.resolve()
            try:
                resolved.relative_to(REPO)
            except ValueError as exc:
                raise SystemExit(f"Source entry escapes the repository: {entry}") from exc
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO prediction on images.")
    parser.add_argument("--weights", required=True, help="Path to trained weights, such as best.pt.")
    parser.add_argument("--source", required=True, help="Image, folder, or video source.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--device", default="cpu", help="Inference device, such as cpu or 0.")
    parser.add_argument("--project", default="outputs/predictions", help="Prediction output directory.")
    parser.add_argument("--name", default="candidate-review", help="Prediction run name.")
    args = parser.parse_args()

    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be positive")
    if not 0 <= args.conf <= 1:
        raise SystemExit("--conf must be between 0 and 1")

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

    weights = repo_path(args.weights, must_be_file=True)
    source = validate_source(repo_path(args.source, must_exist=True))
    project = repo_path(args.project)
    name = safe_run_name(args.name)
    repo_path(project / name)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: python -m pip install ultralytics"
        ) from exc

    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(project),
        name=name,
        save=True,
        save_txt=True,
        save_conf=True,
    )


if __name__ == "__main__":
    main()
