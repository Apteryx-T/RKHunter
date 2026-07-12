# Windows Environment Setup

## Current local environment

The verified project environment is:

```text
Python       3.12.10, 64-bit
virtualenv   D:\RKHunter\.venv
torch        2.13.0+cpu
torchvision  0.28.0+cpu
Ultralytics  8.4.92
CUDA         false
```

This machine is CPU-only. That is sufficient for the current small feasibility, annotation, and smoke-test workflows; it is not intended for large detector training.

Use the virtual-environment interpreter explicitly so Windows Store Python aliases or a different global Python cannot be selected accidentally:

```powershell
D:\RKHunter\.venv\Scripts\python.exe --version
D:\RKHunter\.venv\Scripts\python.exe -m pip --version
```

## Recreate the environment if needed

Install Python 3.11 or 3.12 64-bit from the official Python Windows installer, then run from `D:\RKHunter`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-classifier.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-annotator.txt
```

Dataset files, virtual environments, model weights, caches, training runs, and annotation databases remain local and are ignored by Git.

## Verify the code and annotation environment

```powershell
D:\RKHunter\.venv\Scripts\python.exe -m unittest discover -s tests -v
D:\RKHunter\.venv\Scripts\python.exe scripts\validate_annotation_pipeline.py
```

The second command uses only the explicit local model and local dataset. The annotation launcher, training script, and prediction script force `YOLO_OFFLINE=true`, `YOLO_AUTOINSTALL=false`, and repository-local cache directories before importing Ultralytics.

## Classification baseline

Rebuild and train the original classifier baseline with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_visual_baseline_classification_dataset.ps1
D:\RKHunter\.venv\Scripts\python.exe scripts\train_classifier.py --epochs 10
```

The classifier is only a small visual baseline. Drone detection still requires reviewed bounding boxes and a detector.

## Local annotation tool

See [annotation-tool.md](annotation-tool.md) for startup, review, model-upgrade, export, and validation instructions.
