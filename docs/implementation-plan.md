# RKHunter Implementation Plan

## Phase 1: Dataset Feasibility

- Define target terrain: Gobi, desert, dry lake bed, salt flat, and barren gravel fields.
- Build a seed dataset with meteorite references, barren-ground backgrounds, and rock/metal distractors.
- Clean irrelevant images before labeling.
- Label suspicious targets rather than confirmed meteorites.

## Phase 2: First Detector

- Train a small object detector on cleaned and labeled images.
- Optimize for recall first. Missing a real candidate is worse than producing extra field checks.
- Track false positives carefully and add them to the distractor dataset.

## Phase 3: Simulation

- Use a phone from elevated angles to simulate aerial imagery.
- Place dark stones, metal pieces, ordinary rocks, and shadow distractors on sand or gravel.
- Test whether the detector can recover candidate locations from wide images.

## Phase 4: Drone Trial

- Rent or borrow a drone before buying one.
- Fly a small grid route over a legal test area.
- Compare AI-detected candidate coordinates with manual inspection.

## Phase 5: Field Workflow

- First pass: wide-area scan.
- Second pass: low-altitude close-up capture for candidates.
- Ground pass: GPS-guided human verification.
- Lab pass: only promising samples need mineralogical confirmation.
