$ErrorActionPreference = "Stop"

$repo = "D:\RKHunter"
$experiment = Join-Path $repo "experiments\visual-baseline-001"
$reviewedCsv = Join-Path $experiment "output\review_sheet_reviewed.csv"
$datasetRoot = Join-Path $repo "data\processed\visual-baseline-001"
$classificationRoot = Join-Path $datasetRoot "classification"
$candidateRoot = Join-Path $datasetRoot "candidates"
$manifestPath = Join-Path $datasetRoot "manifest.csv"
$summaryPath = Join-Path $datasetRoot "summary.csv"
$readmePath = Join-Path $datasetRoot "README.md"

if (-not (Test-Path $reviewedCsv)) {
  throw "Reviewed CSV not found: $reviewedCsv"
}

function Normalize-ClassName {
  param([string]$Category)
  switch ($Category) {
    "meteorite_reference" { return "meteorite" }
    "background" { return "background" }
    "distractor" { return "distractor" }
    default { return $Category }
  }
}

function Get-Split {
  param(
    [int]$Index,
    [int]$Total
  )

  if ($Total -le 1) {
    return "train"
  }

  $valCount = [Math]::Max(1, [Math]::Floor($Total * 0.2))
  $testCount = [Math]::Floor($Total * 0.1)

  if ($Total -lt 8) {
    $testCount = 0
  }

  $trainCount = $Total - $valCount - $testCount
  if ($trainCount -lt 1) {
    $trainCount = 1
    $valCount = [Math]::Max(0, $Total - $trainCount)
    $testCount = 0
  }

  if ($Index -lt $trainCount) {
    return "train"
  }
  if ($Index -lt ($trainCount + $valCount)) {
    return "val"
  }
  return "test"
}

function Copy-ImageRecord {
  param(
    [object]$Row,
    [string]$DestinationRoot,
    [string]$Subset,
    [string]$ClassName,
    [string]$Decision
  )

  $source = Join-Path $repo $Row.file
  if (-not (Test-Path $source)) {
    throw "Source image not found: $source"
  }

  $destDir = Join-Path $DestinationRoot (Join-Path $Subset $ClassName)
  New-Item -ItemType Directory -Force -Path $destDir | Out-Null

  $baseName = [System.IO.Path]::GetFileNameWithoutExtension($source)
  $ext = [System.IO.Path]::GetExtension($source)
  $safeName = "{0}_{1}{2}" -f $Decision, $baseName, $ext
  $dest = Join-Path $destDir $safeName
  Copy-Item -Path $source -Destination $dest -Force

  return $dest
}

$rows = @(Import-Csv $reviewedCsv)
$kept = @($rows | Where-Object { $_.human_decision -eq "keep" })
$maybe = @($rows | Where-Object { $_.human_decision -eq "maybe" })

New-Item -ItemType Directory -Force -Path $classificationRoot | Out-Null
New-Item -ItemType Directory -Force -Path $candidateRoot | Out-Null

$manifest = @()

foreach ($group in ($kept | Group-Object category)) {
  $items = @($group.Group | Sort-Object file)
  $className = Normalize-ClassName $group.Name
  for ($i = 0; $i -lt $items.Count; $i++) {
    $split = Get-Split -Index $i -Total $items.Count
    $dest = Copy-ImageRecord -Row $items[$i] -DestinationRoot (Join-Path $classificationRoot "images") -Subset $split -ClassName $className -Decision "keep"
    $manifest += [pscustomobject]@{
      source_file = $items[$i].file
      output_file = $dest.Replace($repo + "\", "")
      dataset = "classification"
      split = $split
      class = $className
      decision = "keep"
      reviewer_notes = $items[$i].reviewer_notes
    }
  }
}

foreach ($group in ($maybe | Group-Object category)) {
  $items = @($group.Group | Sort-Object file)
  $className = Normalize-ClassName $group.Name
  foreach ($item in $items) {
    $dest = Copy-ImageRecord -Row $item -DestinationRoot $candidateRoot -Subset "maybe" -ClassName $className -Decision "maybe"
    $manifest += [pscustomobject]@{
      source_file = $item.file
      output_file = $dest.Replace($repo + "\", "")
      dataset = "candidate"
      split = "maybe"
      class = $className
      decision = "maybe"
      reviewer_notes = $item.reviewer_notes
    }
  }
}

$manifest | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $manifestPath

$summary = $manifest |
  Group-Object dataset, split, class, decision |
  ForEach-Object {
    [pscustomobject]@{
      dataset = $_.Group[0].dataset
      split = $_.Group[0].split
      class = $_.Group[0].class
      decision = $_.Group[0].decision
      count = $_.Count
    }
  } |
  Sort-Object dataset, split, class, decision
$summary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryPath

$readme = @"
# Visual Baseline 001 Dataset

This local dataset was built from `experiments/visual-baseline-001/output/review_sheet_reviewed.csv`.

It is intentionally kept out of Git because project data folders are ignored by `.gitignore`.

## Contents

- `classification/images/train|val|test/background`
- `classification/images/train|val|test/distractor`
- `classification/images/train|val|test/meteorite`
- `candidates/maybe/*`
- `manifest.csv`
- `summary.csv`

## Important

This is an image-classification seed dataset, not a YOLO detection dataset.

YOLO detection training requires bounding-box labels around meteorite candidates. The current positive meteorite images are whole-specimen or close-up references, so they are useful for appearance learning and manual annotation practice, but not enough for final drone detection.

## Next Step

Use this dataset for a small classification baseline, then create bounding boxes for true detection samples.
"@

Set-Content -Path $readmePath -Value $readme -Encoding UTF8

Write-Host "Dataset root: $datasetRoot"
Write-Host "Manifest: $manifestPath"
Write-Host "Summary: $summaryPath"
