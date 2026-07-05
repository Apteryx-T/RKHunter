from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def count_images(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            parent = path.parent.relative_to(root)
            counts[str(parent)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize image counts under a dataset folder.")
    parser.add_argument("root", type=Path, help="Dataset root to scan.")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Dataset root does not exist: {root}")

    counts = count_images(root)
    total = sum(counts.values())

    print(f"Dataset: {root}")
    print(f"Total images: {total}")
    print()

    for folder, count in sorted(counts.items()):
        print(f"{folder}: {count}")


if __name__ == "__main__":
    main()
