# Visual Baseline 001 Classification Dataset

This note describes the first local classification dataset built from the reviewed visual-baseline sample.

## Local Dataset

Dataset root:

```text
data/processed/visual-baseline-001/
```

Main files:

```text
data/processed/visual-baseline-001/manifest.csv
data/processed/visual-baseline-001/summary.csv
data/processed/visual-baseline-001/README.md
```

Classification images:

```text
data/processed/visual-baseline-001/classification/images/train/
data/processed/visual-baseline-001/classification/images/val/
data/processed/visual-baseline-001/classification/images/test/
```

Candidate images:

```text
data/processed/visual-baseline-001/candidates/maybe/
```

## Current Counts

Reviewed images:

```text
keep:   41
maybe:  14
reject: 10
```

Classification split:

```text
train/background: 14
train/distractor: 13
train/meteorite:  3
val/background:   4
val/distractor:   3
val/meteorite:    1
test/background:  2
test/distractor:  1
```

Candidates:

```text
maybe/background: 9
maybe/distractor: 4
maybe/meteorite:  1
```

## Training

This is a small image-classification baseline, not a YOLO object-detection dataset.

Rebuild the local dataset:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_visual_baseline_classification_dataset.ps1
```

Install optional training dependencies:

```powershell
python -m pip install -r requirements-classifier.txt
```

Train:

```powershell
python scripts\train_classifier.py --epochs 10
```

Predict:

```powershell
python scripts\predict_classifier.py path\to\image_or_folder
```

## Next Step

After the classification baseline is working, create bounding-box labels for true detection samples. YOLO training should only start after the positive images have object boxes.
