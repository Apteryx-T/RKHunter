# Annotation Guide

## Main Principle

Label visual candidates, not proven meteorites. Final confirmation requires field inspection and laboratory checks.

## Classes

### suspected_meteorite

Use for a rock-like target that visually stands out from the terrain and could be worth field inspection.

Typical cues:

- Dark exterior against pale sand, salt flat, or gravel
- Rounded or irregular rock shape
- Possible fusion-crust-like surface
- Unusual isolated position

### dark_rock

Use for ordinary dark stones, basalt, volcanic rocks, or rock fragments that could confuse the detector.

### metal_debris

Use for visible artificial metallic objects or bright reflective debris.

### shadow

Use for compact shadows that could be mistaken for dark rocks.

### background

Use only if your annotation tool requires an explicit background class. Most YOLO workflows do not need boxes for background.

## Box Rules

- Draw tight boxes around visible targets.
- Do not include large amounts of surrounding terrain.
- Skip objects that are too blurry to classify.
- If unsure between `suspected_meteorite` and `dark_rock`, use `dark_rock` unless it is a strong candidate.

## First Dataset Size Target

For the first training run, aim for:

- 80-150 useful images
- 150-400 total boxes
- A separate validation set with terrain that is visually different from training images
