from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def repository_path(value: str) -> Path:
    path = Path(value)
    resolved = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(REPO)
    except ValueError as exc:
        raise SystemExit(f"Path must stay inside the repository: {resolved}") from exc
    return resolved


def acquire_database_lock(database_path: Path):
    lock_path = database_path.with_suffix(database_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise SystemExit(
            f"Another annotation server is already using database: {database_path}"
        ) from exc
    return handle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local RKHunter annotation tool.")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback host only.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--database", default="experiments/annotation-tool/annotator-v1.db"
    )
    parser.add_argument("--dataset", help="YOLO dataset root to register and import.")
    parser.add_argument("--project-id", default="rkhunter-yolo")
    parser.add_argument("--project-name", default="RKHunter YOLO Review")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--label-dir", default="labels")
    parser.add_argument(
        "--classes-json",
        default='{"0":"suspected_meteorite","1":"dark_rock","2":"metal_debris","3":"shadow","4":"background"}',
    )
    parser.add_argument("--imported-status", default="auto_labeled")
    parser.add_argument("--model", help="Explicit local YOLO weights path.")
    parser.add_argument("--model-id", default="rkhunter-yolo-current")
    parser.add_argument("--model-name", default="RKHunter YOLO current")
    parser.add_argument("--no-import", action="store_true")
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("The annotation tool is local-only; --host must be a loopback address.")

    database_path = repository_path(args.database)
    database_lock = acquire_database_lock(database_path)

    cache_root = REPO / "models" / "annotation-tool-cache"
    yolo_cache = cache_root / "ultralytics"
    mpl_cache = cache_root / "matplotlib"
    yolo_cache.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    # Enforce the local/offline boundary even if the parent shell has conflicting values.
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_cache)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"

    from rkhunter.annotator.api import create_app
    from rkhunter.annotator.service import AnnotationService

    service = AnnotationService(REPO, database_path)
    if args.dataset:
        classes = {int(key): value for key, value in json.loads(args.classes_json).items()}
        service.register_project(
            args.project_id,
            args.project_name,
            args.dataset,
            classes,
            image_dir=args.image_dir,
            label_dir=args.label_dir,
        )
        if not args.no_import:
            result = service.import_yolo(
                args.project_id, imported_status=args.imported_status
            )
            print(f"Import: {result}")
    if args.model:
        model = service.register_model(
            args.model_id,
            args.model_name,
            "ultralytics_yolo",
            args.model,
            {
                "conf": 0.05,
                "iou": 0.45,
                "imgsz": 640,
                "max_det": 30,
                "allowed_class_ids": [0, 1, 2, 3],
            },
        )
        print(f"Model: {model['name']} {model['version']} {model['sha256']}")

    app = create_app(service)
    print(f"RKHunter Annotation Tool: http://{args.host}:{args.port}")
    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        database_lock.close()


if __name__ == "__main__":
    main()
