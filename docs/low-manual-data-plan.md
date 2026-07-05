# Low-Manual Data Plan

The goal is to avoid spending weeks manually screening and annotating weak web-image data.

## Strategy

Use authoritative data first, then use AI-assisted pre-screening to reduce manual review.

```text
Official records
  -> confirmed meteorite metadata
  -> museum/reference images
  -> automatic image pre-screening
  -> small human review queue
  -> first training set
  -> false-positive review
```

## Step 1: Build a Confirmed Meteorite Index

Start with the Meteoritical Bulletin Database.

Target fields:

- meteorite name
- class
- group/type
- fall or find
- country/region
- mass
- photograph availability
- source URL

Output:

```text
data/external/authoritative-meteorites/meteorite_index.csv
```

## Step 2: Collect Reference Images

Collect only traceable images from official or museum-like sources.

Each image needs a metadata row:

- local file path
- source URL
- institution
- specimen ID or meteorite name
- license/reuse terms
- access date
- notes

Output:

```text
data/external/museum-images/manifest.csv
```

## Step 3: Automatic Pre-Screening

Use a vision model to group images before manual review.

Suggested buckets:

- likely_meteorite_reference
- likely_rock_distractor
- barren_background
- irrelevant
- needs_human_review

Output:

```text
data/processed/auto-prescreen/
```

## Step 4: Human Review Only the Queue

Do not review every image manually.

Review only:

- high-confidence meteorite references
- ambiguous candidate images
- false positives that look like meteorites

Output:

```text
data/processed/review-queue/
```

## Step 5: Label a Small First Dataset

The first manually labeled dataset should be small.

Target:

- 50-100 high-value images
- 150-300 boxes
- strong false-positive coverage

Recommended labels:

- suspected_meteorite
- dark_rock
- metal_debris
- shadow

## Step 6: Add Mission-Like Images Later

Museum images are not enough for drone detection. After the reference library is stable, add phone-simulated or drone-like images from barren terrain.

This is where object detection performance will start to become meaningful.
