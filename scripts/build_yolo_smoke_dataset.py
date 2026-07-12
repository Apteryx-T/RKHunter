from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO / "data" / "raw" / "seed-openverse"
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.csv"
DEFAULT_OUTPUT = REPO / "data" / "processed" / "rkhunter"

GROUPS = {
    "meteorite_reference": (0, "suspected_meteorite"),
    "distractor_rocks": (1, "dark_rock"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_index(index: int, total: int) -> str:
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.20)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def load_source_rows() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in GROUPS}
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            group = row["group"]
            if group not in grouped:
                continue
            filename = Path(row["file"]).name
            source = SOURCE_ROOT / group / filename
            if source.exists():
                row["resolved_source_file"] = str(source.relative_to(REPO))
                grouped[group].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a license-traceable weak-label YOLO smoke-test dataset."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=75)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite non-empty dataset: {output}. "
            "Move it aside or choose --output."
        )

    grouped = load_source_rows()
    rng = random.Random(args.seed)
    selected: list[tuple[str, dict[str, str]]] = []
    for group, rows in grouped.items():
        if len(rows) < args.per_class:
            raise SystemExit(
                f"Not enough source images for {group}: {len(rows)} < {args.per_class}"
            )
        rows = sorted(rows, key=lambda row: row["resolved_source_file"])
        rng.shuffle(rows)
        chosen = rows[: args.per_class]
        chosen.sort(key=lambda row: row["resolved_source_file"])
        selected.extend((group, row) for row in chosen)

    manifest_rows: list[dict[str, str | int]] = []
    for group, (class_id, class_name) in GROUPS.items():
        class_rows = [row for row_group, row in selected if row_group == group]
        for index, row in enumerate(class_rows):
            split = split_for_index(index, len(class_rows))
            source = REPO / row["resolved_source_file"]
            try:
                with Image.open(source) as image:
                    image.verify()
                with Image.open(source) as image:
                    width, height = image.size
            except Exception as exc:
                raise SystemExit(f"Invalid source image {source}: {exc}") from exc

            stem = f"{class_id}_{source.stem}"
            image_dest = output / "images" / split / f"{stem}{source.suffix.lower()}"
            label_dest = output / "labels" / split / f"{stem}.txt"
            image_dest.parent.mkdir(parents=True, exist_ok=True)
            label_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, image_dest)

            # Weak label for engineering smoke tests only. It is deliberately broad
            # because these seed images were collected for classification, not detection.
            label_dest.write_text(f"{class_id} 0.500000 0.500000 0.800000 0.800000\n", encoding="utf-8")

            manifest_rows.append(
                {
                    "source_file": row["resolved_source_file"],
                    "output_image": str(image_dest.relative_to(REPO)),
                    "output_label": str(label_dest.relative_to(REPO)),
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "annotation_method": "weak_center_box",
                    "width": width,
                    "height": height,
                    "sha256": sha256(source),
                    "title": row.get("title", ""),
                    "creator": row.get("creator", ""),
                    "source": row.get("source", ""),
                    "source_page_url": row.get("foreign_landing_url", ""),
                    "license": row.get("license", ""),
                    "license_version": row.get("license_version", ""),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    counts = Counter((str(row["split"]), str(row["class_name"])) for row in manifest_rows)
    summary_path = output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "class_name", "images", "boxes"])
        for (split, class_name), count in sorted(counts.items()):
            writer.writerow([split, class_name, count, count])

    readme = output / "README.md"
    readme.write_text(
        "# RKHunter YOLO Smoke Dataset\n\n"
        "This dataset is for engineering validation only. It is built from the "
        "Openverse classification seed images and uses broad `weak_center_box` labels.\n\n"
        "It is not a field-ready detector dataset and must not be used as evidence "
        "that the model can find meteorites in drone imagery. Replace weak labels "
        "with human-reviewed boxes and add clean mission-like imagery before any "
        "real evaluation.\n",
        encoding="utf-8",
    )

    print(f"Dataset: {output}")
    print(f"Images: {len(manifest_rows)}")
    print(f"Boxes: {len(manifest_rows)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
