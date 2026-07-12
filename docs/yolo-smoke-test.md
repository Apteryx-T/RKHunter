# YOLO Smoke Test Dataset

The local dataset at `data/processed/rkhunter/` is an engineering smoke-test dataset, not a field-ready meteorite detector dataset.

## Build

```powershell
D:\RKHunter\.venv\Scripts\python.exe scripts\build_yolo_smoke_dataset.py
```

The builder selects 75 license-traceable Openverse meteorite-reference images and 75 rock-distractor images. It creates deterministic train, validation, and test splits with one broad YOLO box per image.

## Annotation limitation

Every box is marked `weak_center_box` in the generated manifest. These are broad engineering labels derived from classification images, not human-reviewed object boxes. They are sufficient to exercise parsing, training, checkpoint saving, and inference, but not to measure real detection quality.

Do not claim performance on drone imagery from this dataset. Replace weak boxes with human-reviewed annotations and add clean top-down natural-ground positives before any field evaluation.

## Install

```powershell
D:\RKHunter\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
```

## CPU smoke training with explicit local weights

```powershell
D:\RKHunter\.venv\Scripts\python.exe scripts\train_yolo.py `
  --data configs\dataset.yaml `
  --model models\yolov8n.pt `
  --epochs 5 `
  --imgsz 320 `
  --batch-size 16 `
  --workers 0 `
  --device cpu `
  --project experiments\yolo `
  --name smoke-weak-labels-pretrained-5e
```

The training script requires an existing local file and forces Ultralytics offline mode. It will not resolve a model name by downloading weights.

## Verified local result

Environment:

- Python 3.12.10
- torch 2.13.0+cpu
- ultralytics 8.4.92
- CPU: Intel Core i7-4720HQ

The offline, randomly initialized 5-epoch run completed successfully but produced zero validation detections. This confirms the offline training path, not model quality.

The official pretrained `yolov8n.pt` was downloaded through the direct Ultralytics GitHub release URL and stored locally under the ignored `models/` directory. A 10-epoch transfer-learning smoke run completed successfully. Its weak-label validation metrics were:

- precision: 0.787
- recall: 0.767
- mAP50: 0.900
- mAP50-95: 0.801

These numbers measure agreement with broad weak center boxes. They are not meaningful field-detection metrics.

## Mission-like holdout check

Four separately sourced in-situ meteorite images were excluded from training and used as a qualitative domain check. At confidence 0.05:

- Rub' al-Khali desert image: no detection
- Mojave desert image: two `dark_rock` detections
- Antarctic image with field equipment: one `dark_rock` detection
- clean Antarctic image: one `suspected_meteorite` and one `dark_rock` detection

Visual inspection showed oversized and unreliable boxes, including a box covering most of the clean Antarctic scene. This confirms that the smoke-test model learned the weak-box construction and source-image biases. It must not be treated as a usable meteorite detector.

## Next data milestone

The engineering pipeline is ready. The next milestone is data quality rather than more smoke training:

- manually review and tighten boxes;
- add background-only desert images;
- add small top-down rocks on natural terrain;
- keep source-series duplicates in the same split;
- reserve a mission-like holdout set that is never used for training.
