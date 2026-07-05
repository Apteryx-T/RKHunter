# RKHunter

RKHunter is an early-stage project for using aerial imagery and AI-assisted detection to find meteorite candidates in desert, Gobi, dry lake bed, and other barren-ground environments.

The goal is not to let AI directly prove that a rock is a meteorite. The goal is to scan imagery, rank suspicious targets, export coordinates, and support human field verification.

## Project Status

Current phase: dataset and feasibility validation.

No drone hardware is required for the first phase. The recommended first milestone is to build a small image dataset, clean it, label candidate objects, and train a simple object detector.

## Repository Structure

```text
RKHunter/
  assets/                 Project images and lightweight visual assets
  configs/                Dataset and model configuration files
  data/                   Local datasets, ignored by Git by default
    raw/                  Original downloaded or captured images
    processed/            Cleaned and split datasets
    external/             Third-party datasets or exports
  docs/                   Project plans, data notes, and field protocols
  experiments/            Training runs and experiment outputs, ignored by Git
  models/                 Model weights, ignored by Git
  notebooks/              Research notebooks
  outputs/                Detection outputs and reports, ignored by Git
  scripts/                Command-line scripts for data, training, and inference
  src/rkhunter/           Python package source code
  tests/                  Automated tests
  tools/                  Helper tools and utilities
```

## Suggested First Milestones

1. Collect and clean 200-500 seed images.
2. Label candidate objects with CVAT, Label Studio, or Roboflow.
3. Train a small YOLO-style object detector.
4. Review false positives and expand the distractor dataset.
5. Run a phone-based simulation before renting or buying a drone.

## Suggested Labels

- `suspected_meteorite`
- `dark_rock`
- `metal_debris`
- `shadow`
- `background`

## Data Policy

Large images, datasets, model weights, and experiment outputs are ignored by Git by default. Keep only metadata, documentation, sample configs, and lightweight assets in the repository.

Before publishing third-party images or datasets, check each source license and attribution requirement.
