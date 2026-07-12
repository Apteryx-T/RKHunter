# RKHunter Local Annotation Tool

The annotation tool is a local-first review system for model-assisted bounding boxes. It connects YOLO import, automatic proposals, Canvas correction, human approval, audit history, immutable exports, and later model upgrades.

## Current verified state

- Tool version: `0.2.0`
- Database schema: `2`
- Runtime: Python 3.12, CPU-only PyTorch and Ultralytics
- Current local dataset: 150 images (`train` 104, `val` 30, `test` 16)
- Browser, API, real local-model inference, export, and Ultralytics label parsing have been exercised locally.

This verifies the workflow, not detector quality. The current detector was trained from broad weak boxes and is not evidence of real drone-image performance.

## Safety boundary

- The server accepts loopback hosts only and normally binds to `127.0.0.1`.
- Database, dataset, model, cache, and export paths must resolve inside the repository.
- The launcher forcibly enables Ultralytics offline mode and disables auto-install.
- Models must already exist locally; model names that could trigger an implicit download are not accepted by the training or prediction scripts.
- Automatic proposals are always drafts. Reviewed or rejected images, including reviewed empty backgrounds, cannot be overwritten by a model.
- Image edits use conditional revisions, so stale clients receive HTTP `409` instead of overwriting newer work.
- Source image or source-label changes are detected during re-import and require an explicit migration instead of silently changing reviewed data.

## Install

Use the project virtual environment:

```powershell
D:\RKHunter\.venv\Scripts\python.exe -m pip install -r requirements-annotator.txt
D:\RKHunter\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
```

The first file installs the web/API and Pillow dependencies. The second installs the optional Ultralytics adapter.

## Start

```powershell
D:\RKHunter\.venv\Scripts\python.exe scripts\run_annotation_tool.py `
  --dataset data\processed\rkhunter `
  --project-id rkhunter-yolo `
  --project-name "RKHunter YOLO Review" `
  --model experiments\yolo\smoke-weak-labels-pretrained-10e\weights\best.pt
```

Open `http://127.0.0.1:8765`.

Runtime state is local and ignored by Git:

```text
experiments/annotation-tool/annotator-v1.db
experiments/annotation-tool/exports/
models/annotation-tool-cache/
```

The filename retains `v1` for backward compatibility; the database itself migrates transactionally and currently reports schema `2`. A pre-migration backup of the working database should be retained when upgrading production data.

## Review workflow

1. Select an image from the queue.
2. Run the current local model on one image or create a persistent batch job.
3. Draw, move, resize, delete, or reclassify boxes in the Canvas editor.
4. Save and approve the image, confirm an empty background, or reject the image.
5. Export reviewed work to a new YOLO revision.

Imported weak labels start as `auto_labeled` drafts. A background is an approved image with an empty annotation list and empty `.txt` label; a `background` box is rejected at import and save boundaries.

Batch jobs pin an immutable model revision, persist progress in SQLite, and can recover an interrupted `running` job after a process restart. The browser loads the complete queue in 500-row pages instead of silently stopping at the first page.

Run only one annotation-server process against a given database. The launcher holds an operating-system file lock and refuses a second process before it can claim recovery work.

## Immutable model upgrades

The model registry separates a mutable model alias from immutable revisions. A revision fingerprints:

- adapter name;
- RKHunter tool version and adapter implementation version;
- resolved local model path;
- model file or directory content SHA256;
- canonical adapter configuration.

Every model-generated annotation and batch run records `model_revision_id`. Re-registering the same alias with new weights or configuration switches future work to a new revision without rewriting historical provenance. If registered model content changes in place, inference is refused until the new revision is registered.

Built-in adapters are:

- `ultralytics_yolo`, operational with an explicit local `.pt` file;
- `grounding_dino`, optional and restricted to an explicitly downloaded local Transformers directory with `local_files_only=True`.

Third-party adapters can use the `rkhunter.annotator_adapters` Python entry-point group.
They should expose a stable `adapter_version` attribute and bump it whenever inference behavior changes.

## YOLO export rules

Each export is written to a hidden staging directory, validated, and atomically renamed to a unique revision. Failed exports leave no published partial revision. Images are copied rather than hard-linked, nested paths are preserved, Windows case-insensitive collisions are rejected, and source image hashes are checked.

Training exports are reviewed-only. Draft export is deliberately disabled because an unreviewed image with no approved box must not become a false background.

The exported `dataset.yaml` is relative to its own directory and declares only non-empty splits. The `background` UI class is removed from detector names; `manifest.json` records the source-to-export class mapping, split counts, box counts, and readiness information.

`train_ready` requires:

- at least one reviewed train image;
- at least one approved object box in train;
- at least one reviewed val image;
- at least one approved object box in val.

A partial revision is still useful as an immutable review snapshot, but `scripts/train_yolo.py` refuses it until the manifest is train-ready.

`train_ready` is still only a structural minimum. A meaningful validation split needs representative human-reviewed positives and backgrounds. The test split remains independent and is never substituted for validation during model selection.

The manifest stores SHA256 for every canonical image, label, and `dataset.yaml`. The training script verifies those hashes and copies only canonical files into an ignored training-work directory before Ultralytics runs. Ultralytics `*.cache` files are therefore written beside the working copy, not into the published revision.

## Validate the complete local chain

This command creates an isolated ignored database and export, runs the real local model on CPU, parses the result through Ultralytics, and verifies that source image sizes and timestamps did not change:

```powershell
D:\RKHunter\.venv\Scripts\python.exe scripts\validate_annotation_pipeline.py
```

Results are written under:

```text
experiments/annotation-tool/validation/<timestamp>/summary.json
```

The validation script approves one model proposal, one weak positive validation label, and one empty test image only inside its isolated validation database. Ultralytics parses a separate working copy, so the canonical export is unchanged. These artifacts are pipeline tests, not human ground truth.

## Local reviewed smoke checkpoint (2026-07-12)

An agent-assisted visual pass approved or confirmed empty backgrounds for 16 additional reference images. Together with the existing reviewed test image, the immutable export contains:

```text
train: 9 images, 6 boxes, 3 empty backgrounds
val:   7 images, 6 boxes, 1 empty background
test:  1 image,  1 box
```

The local export revision is `rkhunter-yolo-20260712T063557Z-ca5b55`. A five-epoch CPU smoke run is under `experiments/yolo/reviewed-annotations-v1-5e/`, and its best weights are registered as the separate experimental model alias `rkhunter-yolo-reviewed-v1`. The audit actor is `codex-visual-review`; this checkpoint must not be presented as human ground truth.

Validation reported `mAP50=0.912` and `mAP50-95=0.484`, but the validation set has only seven images. Qualitative test predictions still contain broad duplicate boxes and meteorite false positives on terrestrial rocks. The remaining draft queue was therefore not overwritten in bulk. The next useful work is more real human review and substantially more top-down natural-ground positives and hard-negative terrain.

## Train an approved revision

After enough real human review makes an export train-ready:

```powershell
D:\RKHunter\.venv\Scripts\python.exe scripts\train_yolo.py `
  --data experiments\annotation-tool\exports\<revision>\dataset.yaml `
  --model models\yolov8n.pt `
  --device cpu `
  --epochs 10
```

All training inputs must already exist locally. Outputs remain under ignored `experiments/` directories.

## Licensing note

The installed Ultralytics package reports the AGPL-3.0 license. Review licensing before distributing this tool as closed-source or commercial software.
