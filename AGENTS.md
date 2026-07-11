# RKHunter Agent Notes

## Project Purpose

RKHunter is an early-stage project for AI-assisted meteorite candidate discovery from aerial imagery. The target environments are desert, Gobi, dry lake bed, and other barren-ground regions.

The goal is not to prove a rock is a meteorite automatically. The system should eventually scan imagery, rank suspicious targets, export candidate coordinates, and support human field verification.

## Current Phase

Current phase: dataset and feasibility validation.

Drone hardware is not required yet. The project is currently validating whether a small visual dataset and a lightweight model workflow can run locally.

## Repository Location

Local project root:

```text
D:\RKHunter
```

GitHub remote:

```text
https://github.com/Apteryx-T/RKHunter.git
```

Default branch:

```text
main
```

## Important Data Policy

Large data and generated artifacts are intentionally ignored by Git:

- `data/raw/*`
- `data/processed/*`
- `data/external/*`
- `experiments/*`
- `outputs/*`
- `models/*`
- `.venv/`

Do not force-add these unless the user explicitly asks. GitHub should contain code, configs, and documentation; datasets and models remain local.

## Current Local Data State

Seed image work exists locally under:

```text
data/raw/seed-openverse/
```

Known seed groups:

```text
meteorite_reference: 80
background/desert aerial: 80
distractor rocks: 80
```

Authoritative meteorite metadata and image-source workflow exists locally under:

```text
data/external/authoritative-meteorites/
data/external/museum-images/reviewed-reference-images/
```

The reviewed museum/reference images produced a small useful reference set. First visual experiment data exists under:

```text
experiments/visual-baseline-001/
```

## Visual Baseline 001

A first manual/agent review has been completed for 65 images:

```text
keep:   41
maybe:  14
reject: 10
```

Reviewed outputs:

```text
experiments/visual-baseline-001/output/review_sheet_reviewed.csv
experiments/visual-baseline-001/output/visual-baseline-001-decisions.csv
experiments/visual-baseline-001/output/review_decision_summary.csv
```

A local classification dataset has been built at:

```text
data/processed/visual-baseline-001/
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

Candidate/maybe split:

```text
maybe/background: 9
maybe/distractor: 4
maybe/meteorite:  1
```

This is an image-classification seed dataset, not a YOLO object-detection dataset.

## Python Environment

Python 3.12.10 was installed locally.

Project virtual environment:

```text
D:\RKHunter\.venv
```

The classifier dependencies were installed into the virtual environment:

```text
torch
torchvision
pillow
```

Verified environment:

```text
torch 2.13.0+cpu
torchvision 0.28.0+cpu
cuda False
```

This is CPU-only. It is acceptable for the small first baseline.

## First Classifier Training

A first 5-epoch classifier run completed successfully using:

```powershell
D:\RKHunter\.venv\Scripts\python.exe scripts\train_classifier.py --epochs 5 --batch-size 4
```

Best validation accuracy observed:

```text
Best val_acc: 0.875
```

Local model output:

```text
models/classifier-visual-baseline-001.pt
```

The model file is ignored by Git and should stay local unless the user explicitly requests model publishing.

Prediction script was verified on:

```text
data/processed/visual-baseline-001/classification/images/test
data/processed/visual-baseline-001/candidates/maybe
```

Important observation: the `maybe` Tissint small meteorite image was predicted as `distractor`, which confirms that the current positive meteorite set is too small and not drone-like enough. The workflow works, but the dataset is not yet sufficient for real detection.

## Key Scripts

Classification dataset rebuild:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_visual_baseline_classification_dataset.ps1
```

Train classifier:

```powershell
D:\RKHunter\.venv\Scripts\python.exe scripts\train_classifier.py --epochs 10
```

Predict with classifier:

```powershell
D:\RKHunter\.venv\Scripts\python.exe scripts\predict_classifier.py path\to\image_or_folder
```

YOLO scripts exist but should not be treated as ready for final detector training until bounding-box labels are created:

```text
scripts/train_yolo.py
scripts/predict_yolo.py
configs/dataset.yaml
```

MetBull metadata preparation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_metbull_index.ps1 -InputCsv data\external\authoritative-meteorites\manual-downloads\YOUR_FILE.csv
```

## Current Git State At Last Summary

Latest pushed commit at time of this note:

```text
914c4f0 Add visual baseline classifier workflow
```

Before editing, always check:

```powershell
git status -sb
```

User preference: work locally first, avoid frequent GitHub pushes unless the user asks. The user explicitly asked to push at previous milestones, and the repository was synchronized then.

## Recommended Next Work

1. Expand positive meteorite data.
   - Priority: meteorites or meteorite-like candidates on natural ground, not only museum close-ups.
   - Need more examples with desert/Gobi/dry-lake context.

2. Start bounding-box annotation.
   - Create a small YOLO-style detection set only after marking candidate rock locations.
   - Suggested labels remain in `configs/dataset.yaml`: `suspected_meteorite`, `dark_rock`, `metal_debris`, `shadow`, `background`.

3. Improve the local review loop.
   - Keep using `experiments/visual-baseline-001/output/review_tool.html` for manual review.
   - Merge user decisions into CSVs before rebuilding datasets.

4. Run a second classifier baseline after adding more positive samples.
   - The current classifier is proof of pipeline, not a reliable model.

5. Only after bounding boxes exist, train YOLO.

## Agent Cautions

- Do not claim the current model can find meteorites in real drone imagery.
- Do not upload data, model weights, or `.venv` to GitHub.
- Do not overwrite local datasets without checking existing manifests and summaries.
- Prefer reproducible scripts over one-off manual file moves.
- Use `D:\RKHunter\.venv\Scripts\python.exe` for Python commands in this project.
- If adding data, update local manifests/summaries and document the source/license.
