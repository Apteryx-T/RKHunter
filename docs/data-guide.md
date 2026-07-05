# Data Guide

## Recommended Data Groups

1. Meteorite reference images
2. Desert and dry lake bed aerial backgrounds
3. Distractor rocks, including basalt, slag, iron ore, volcanic rocks, and ordinary dark stones
4. Simulated field images captured by phone or drone

## Cleaning Rules

Remove:

- People, buildings, signs, logos, exhibition halls, and city aerial images
- Images where the rock is too small or too blurred
- Images dominated by sky, water, vegetation, or unrelated objects

Keep:

- Clear rock or meteorite-like candidates
- Barren terrain backgrounds
- Common false positives such as shadows, dark stones, metal debris, and tire tracks

## Recommended Split

- Train: 70%
- Validation: 20%
- Test: 10%

Keep test data visually different from training data whenever possible.
