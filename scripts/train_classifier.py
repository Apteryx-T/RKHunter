from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small image-classification baseline for RKHunter."
    )
    parser.add_argument(
        "--data",
        default="data/processed/visual-baseline-001/classification/images",
        help="Directory containing train/val/test class folders.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size.")
    parser.add_argument(
        "--output",
        default="models/classifier-visual-baseline-001.pt",
        help="Output model path.",
    )
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise SystemExit(
            "PyTorch and torchvision are required for classifier training.\n"
            "Install them first, for example:\n"
            "  python -m pip install torch torchvision\n"
        ) from exc

    data_root = Path(args.data)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            f"Expected train and val folders under {data_root}. "
            "Build the visual-baseline classification dataset first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tfms = transforms.Compose(
        [
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_data = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_data = datasets.ImageFolder(val_dir, transform=eval_tfms)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / max(train_total, 1)
        print(
            f"epoch={epoch:03d} loss={avg_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_data.classes,
                    "imgsz": args.imgsz,
                    "val_acc": val_acc,
                },
                output_path,
            )

    print(f"Best val_acc: {best_acc:.3f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
