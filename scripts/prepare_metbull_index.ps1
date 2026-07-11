param(
  [Parameter(Mandatory = $true)]
  [string]$InputCsv,

  [string]$OutputCsv = "data/external/authoritative-meteorites/processed/meteorite_index.csv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $InputCsv)) {
  throw "Input CSV not found: $InputCsv"
}

function Get-ColumnValue {
  param(
    [object]$Row,
    [string[]]$Candidates
  )

  foreach ($candidate in $Candidates) {
    $property = $Row.PSObject.Properties | Where-Object {
      $_.Name -ieq $candidate -or $_.Name -match $candidate
    } | Select-Object -First 1

    if ($property -and $null -ne $property.Value) {
      return [string]$property.Value
    }
  }

  return ""
}

function Get-MetBullDetailUrl {
  param([string]$Code)

  if ([string]::IsNullOrWhiteSpace($Code)) {
    return ""
  }

  return "https://www.lpi.usra.edu/meteor/metbull.php?code=$Code"
}

$rows = Import-Csv -Path $InputCsv
$normalized = foreach ($row in $rows) {
  $code = Get-ColumnValue $row @("^Code$")
  $name = Get-ColumnValue $row @("^Name$", "meteorite", "meteorite name")
  $abbrev = Get-ColumnValue $row @("^Abbrev$")
  $status = Get-ColumnValue $row @("^Status$")
  $class = Get-ColumnValue $row @("^Type$", "class", "classification", "recclass")
  $fallFind = Get-ColumnValue $row @("^Fall$", "fall/find", "find", "status")
  $place = Get-ColumnValue $row @("^Place$", "country", "region", "locality")
  $mass = Get-ColumnValue $row @("^Mass$", "mass.*g")
  $year = Get-ColumnValue $row @("^Year$")
  $lat = Get-ColumnValue $row @("^Lat$", "reclat", "latitude")
  $lon = Get-ColumnValue $row @("^Long$", "reclong", "longitude", "lng")
  $metbull = Get-ColumnValue $row @("^MetBull$")
  $antarctic = Get-ColumnValue $row @("^Antarctic$")
  $comment = Get-ColumnValue $row @("^Comment$")
  $photo = Get-ColumnValue $row @("photo", "photograph", "image", "picture")
  $detailUrl = Get-ColumnValue $row @("url", "link", "detail")

  if ([string]::IsNullOrWhiteSpace($detailUrl)) {
    $detailUrl = Get-MetBullDetailUrl $code
  }

  [pscustomobject]@{
    metbull_code = $code
    meteorite_name = $name
    abbrev = $abbrev
    status = $status
    classification = $class
    fall_find = $fallFind
    place = $place
    mass_g = $mass
    year = $year
    latitude = $lat
    longitude = $lon
    metbull_bulletin = $metbull
    antarctic = $antarctic
    photo_field = $photo
    detail_url = $detailUrl
    source = "Meteoritical Bulletin Database"
    source_csv = $InputCsv
    notes = $comment
  }
}

$outputDir = Split-Path -Parent $OutputCsv
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$normalized | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $OutputCsv

Write-Host "Input rows: $($rows.Count)"
Write-Host "Output: $OutputCsv"
