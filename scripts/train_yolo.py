from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for RKHunter.")
    parser.add_argument("--data", default="configs/dataset.yaml", help="Path to YOLO dataset YAML.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model name or path.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size.")
    parser.add_argument("--project", default="experiments/yolo", help="Output project directory.")
    parser.add_argument("--name", default="first-detector", help="Run name.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: python -m pip install ultralytics"
        ) from exc

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset config not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
