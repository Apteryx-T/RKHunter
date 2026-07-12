from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def source_snapshot(image_root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(image_root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in image_root.rglob("*")
        if path.is_file()
    }


def repo_path(value: str) -> Path:
    path = Path(value)
    resolved = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(REPO)
    except ValueError as exc:
        raise SystemExit(f"Path must stay inside the repository: {resolved}") from exc
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the local annotation pipeline without changing source data."
    )
    parser.add_argument("--dataset", default="data/processed/rkhunter")
    parser.add_argument(
        "--model",
        default="experiments/yolo/smoke-weak-labels-pretrained-10e/weights/best.pt",
    )
    parser.add_argument(
        "--output-root", default="experiments/annotation-tool/validation"
    )
    parser.add_argument("--max-probe-images", type=int, default=10)
    args = parser.parse_args()

    cache_root = REPO / "models" / "annotation-tool-cache"
    yolo_cache = cache_root / "ultralytics"
    mpl_cache = cache_root / "matplotlib"
    yolo_cache.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_cache)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"

    from rkhunter.annotator.service import AnnotationService

    dataset = repo_path(args.dataset)
    model_path = repo_path(args.model)
    output_root = repo_path(args.output_root)
    if not dataset.is_dir():
        raise SystemExit(f"Dataset not found: {dataset}")
    if not model_path.is_file():
        raise SystemExit(f"Local model not found: {model_path}")
    if args.max_probe_images < 1:
        raise SystemExit("--max-probe-images must be positive")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = output_root / stamp
    image_root = dataset / "images"
    label_root = dataset / "labels"
    before = {
        "images": source_snapshot(image_root),
        "labels": source_snapshot(label_root),
    }
    classes = {
        0: "suspected_meteorite",
        1: "dark_rock",
        2: "metal_debris",
        3: "shadow",
        4: "background",
    }
    service = AnnotationService(REPO, run_root / "validation.db")
    service.register_project(
        "pipeline-validation", "Pipeline validation", dataset, classes
    )
    imported = service.import_yolo("pipeline-validation")
    model = service.register_model(
        "pipeline-model",
        "Pipeline validation model",
        "ultralytics_yolo",
        model_path,
        {
            "conf": 0.05,
            "iou": 0.45,
            "imgsz": 640,
            "max_det": 30,
            "allowed_class_ids": [0, 1, 2, 3],
        },
    )

    train_items = service.list_images(
        "pipeline-validation", split="train", limit=args.max_probe_images
    )["items"]
    positive = None
    for item in train_items:
        proposed = service.auto_label_image(
            "pipeline-validation",
            item["id"],
            "pipeline-model",
            {"conf": 0.05, "imgsz": 640, "max_det": 30, "device": "cpu"},
        )
        if proposed["annotations"]:
            positive = service.save_annotations(
                "pipeline-validation",
                item["id"],
                proposed["annotations"],
                expected_revision=proposed["revision"],
                status="reviewed",
                actor="pipeline-validator",
            )
            break
    if positive is None:
        raise RuntimeError(
            f"No model proposal was produced in the first {len(train_items)} train images"
        )

    validation_items = service.list_images(
        "pipeline-validation", split="val", limit=1
    )["items"]
    if not validation_items:
        raise RuntimeError("A val image is required for pipeline validation")
    validation_detail = service.get_image(
        "pipeline-validation", validation_items[0]["id"]
    )
    if not validation_detail["annotations"]:
        raise RuntimeError("The validation pipeline needs one positive val label")
    service.save_annotations(
        "pipeline-validation",
        validation_detail["id"],
        validation_detail["annotations"],
        expected_revision=validation_detail["revision"],
        status="reviewed",
        actor="pipeline-validator",
    )

    background_items = service.list_images(
        "pipeline-validation", split="test", limit=1
    )["items"]
    if not background_items:
        raise RuntimeError("A test image is required to validate empty-background export")
    background_detail = service.get_image(
        "pipeline-validation", background_items[0]["id"]
    )
    service.save_annotations(
        "pipeline-validation",
        background_detail["id"],
        [],
        expected_revision=background_detail["revision"],
        status="reviewed",
        actor="pipeline-validator",
    )

    exported = service.export_yolo(
        "pipeline-validation", run_root / "exports", reviewed_only=True
    )
    if not exported["train_ready"]:
        raise RuntimeError(f"Export was not train-ready: {exported['readiness_issues']}")

    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    export_path = Path(exported["path"])
    parser_root = run_root / "ultralytics-parser-work"
    shutil.copytree(export_path, parser_root)
    parsed = check_det_dataset(str(parser_root / "dataset.yaml"), autodownload=False)
    train_dataset = YOLODataset(
        parsed["train"], data=parsed, task="detect", imgsz=640, augment=False, cache=False
    )
    validation_dataset = YOLODataset(
        parsed["val"],
        data=parsed,
        task="detect",
        imgsz=640,
        augment=False,
        cache=False,
    )
    test_dataset = YOLODataset(
        parsed["test"],
        data=parsed,
        task="detect",
        imgsz=640,
        augment=False,
        cache=False,
    )
    parsed_train_boxes = sum(len(label["cls"]) for label in train_dataset.labels)
    parsed_validation_boxes = sum(
        len(label["cls"]) for label in validation_dataset.labels
    )
    parsed_test_boxes = sum(len(label["cls"]) for label in test_dataset.labels)
    if (
        len(train_dataset) != 1
        or parsed_train_boxes < 1
        or len(validation_dataset) != 1
        or parsed_validation_boxes < 1
        or len(test_dataset) != 1
        or parsed_test_boxes != 0
    ):
        raise RuntimeError("Ultralytics did not round-trip the exported images and labels")
    if list(export_path.rglob("*.cache")):
        raise RuntimeError("Ultralytics modified the canonical export revision")
    after = {
        "images": source_snapshot(image_root),
        "labels": source_snapshot(label_root),
    }
    if before != after:
        raise RuntimeError("Source image or label size/timestamp changed during validation")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database_schema_version": service.database.schema_version(),
        "import": imported,
        "model_revision_id": model["revision_id"],
        "positive_image": positive["rel_path"],
        "positive_boxes": len(positive["annotations"]),
        "validation_split": "val",
        "validation_positive_boxes": len(validation_detail["annotations"]),
        "background_split": "test",
        "export": exported,
        "ultralytics_train_path": parsed["train"],
        "ultralytics_val_path": parsed.get("val") or parsed.get("test"),
        "ultralytics_train_images": len(train_dataset),
        "ultralytics_train_boxes": parsed_train_boxes,
        "ultralytics_validation_images": len(validation_dataset),
        "ultralytics_validation_boxes": parsed_validation_boxes,
        "ultralytics_test_images": len(test_dataset),
        "ultralytics_test_boxes": parsed_test_boxes,
        "source_files_unchanged": True,
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
