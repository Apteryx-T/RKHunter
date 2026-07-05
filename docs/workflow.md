# RKHunter Workflow

This workflow keeps the project practical while there is no drone hardware yet.

## 1. Raw Seed Data

The first seed dataset is stored locally at:

```text
data/raw/seed-openverse/
```

This folder is intentionally ignored by Git because it contains third-party images and can grow quickly.

## 2. Manual Cleaning

Create a cleaned folder:

```text
data/processed/cleaned-seed/
```

Keep images that are useful for one of these purposes:

- Clear meteorite-like object references
- Barren desert, Gobi, salt flat, or dry lake bed backgrounds
- False positives such as dark rocks, volcanic rocks, ore, slag, metal debris, and strong shadows

Remove images dominated by:

- People
- Exhibition halls
- Buildings or city aerial views
- Logos, diagrams, or text signs
- Sky, water, vegetation, or unrelated objects

## 3. Annotation

Use CVAT, Label Studio, or Roboflow to label candidate objects.

Recommended labels:

- `suspected_meteorite`
- `dark_rock`
- `metal_debris`
- `shadow`
- `background`

Do not label a rock as confirmed meteorite from imagery alone. Label it as a visual candidate.

## 4. Dataset Export

Export labels in YOLO format and place the result under:

```text
data/processed/rkhunter/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

The matching config is:

```text
configs/dataset.yaml
```

## 5. First Training Run

Install training dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install ultralytics opencv-python pyyaml
```

Run:

```powershell
python scripts\train_yolo.py --data configs\dataset.yaml --model yolov8n.pt --epochs 30 --imgsz 1024
```

The first goal is high recall, not perfect precision.

## 6. Review False Positives

After training, inspect predictions and collect false positives into the distractor set. This loop matters more than model size in the early phase.
