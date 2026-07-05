from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO prediction on images.")
    parser.add_argument("--weights", required=True, help="Path to trained weights, such as best.pt.")
    parser.add_argument("--source", required=True, help="Image, folder, or video source.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--project", default="outputs/predictions", help="Prediction output directory.")
    parser.add_argument("--name", default="candidate-review", help="Prediction run name.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: python -m pip install ultralytics"
        ) from exc

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=True,
        save_conf=True,
    )


if __name__ == "__main__":
    main()
