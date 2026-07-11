# Windows Environment Setup

This project needs a real Python installation before local model training can run.

## Current Symptom

If this command returns nothing useful or opens the Microsoft Store placeholder, Python is not installed correctly:

```powershell
python --version
```

## Install Python

Install Python from the official Windows download page:

```text
https://www.python.org/downloads/windows/
```

Recommended version:

```text
Python 3.11 or Python 3.12, 64-bit
```

During installation, check:

```text
Add python.exe to PATH
```

Then close and reopen PowerShell.

## Verify

```powershell
python --version
python -m pip --version
```

Both commands should print a version.

## Install Classification Dependencies

From the project root:

```powershell
cd D:\RKHunter
python -m pip install -r requirements-classifier.txt
```

## Rebuild The Local Dataset

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_visual_baseline_classification_dataset.ps1
```

## Train The First Baseline

```powershell
python scripts\train_classifier.py --epochs 10
```

The model checkpoint will be saved under:

```text
models/classifier-visual-baseline-001.pt
```

## Predict On Images

```powershell
python scripts\predict_classifier.py path\to\image_or_folder
```

## Note

This classifier is only a small baseline for learning visual differences between meteorite reference images, barren backgrounds, and rock/mineral distractors. Final drone detection still requires bounding-box labels and a detector such as YOLO.
