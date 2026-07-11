from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
      sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run the RKHunter image-classification baseline on images."
    )
    parser.add_argument("images", type=Path, help="Image file or directory to score.")
    parser.add_argument(
        "--model",
        default="models/classifier-visual-baseline-001.pt",
        help="Classifier checkpoint produced by train_classifier.py.",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Override input size.")
    args = parser.parse_args()

    try:
        import torch
        from PIL import Image
        from torchvision import models, transforms
    except ImportError as exc:
        raise SystemExit(
            "PyTorch, torchvision, and Pillow are required for classifier prediction.\n"
            "Install them first, for example:\n"
            "  python -m pip install torch torchvision pillow\n"
        ) from exc

    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        raise SystemExit(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    imgsz = args.imgsz or checkpoint.get("imgsz", 224)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_paths = []
    if args.images.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"):
            image_paths.extend(args.images.rglob(ext))
    else:
        image_paths.append(args.images)

    if not image_paths:
        raise SystemExit(f"No images found under {args.images}")

    with torch.no_grad():
        for image_path in sorted(image_paths):
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0)
            probs = torch.softmax(model(tensor), dim=1)[0]
            score, index = probs.max(dim=0)
            print(f"{image_path}\t{classes[index]}\t{score.item():.4f}")


if __name__ == "__main__":
    main()
