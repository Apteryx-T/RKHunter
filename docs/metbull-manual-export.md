# Meteoritical Bulletin Manual Export

The Meteoritical Bulletin Database is the preferred starting point for confirmed meteorite metadata.

URL:

https://www.lpi.usra.edu/meteor/

## Why Manual Export

The site may require browser JavaScript verification, so direct script scraping can fail. Use the site's own browser export instead of trying to bypass access controls.

## Recommended Steps

1. Open the database in a normal browser.
2. Search or filter for records with photographs.
3. Export the result as CSV.
4. Save the CSV under:

   `data/external/authoritative-meteorites/manual-downloads/`

5. Convert it to the RKHunter standard index:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\prepare_metbull_index.ps1 -InputCsv data\external\authoritative-meteorites\manual-downloads\YOUR_FILE.csv
   ```

6. Review:

   `data/external/authoritative-meteorites/processed/meteorite_index.csv`

## Output Fields

- `meteorite_name`
- `classification`
- `fall_find`
- `country_or_region`
- `mass`
- `year`
- `latitude`
- `longitude`
- `photo_field`
- `detail_url`
- `source`
- `source_csv`
- `notes`

## Important

Metadata can be stored locally. Images should be downloaded only after checking each source's license and reuse terms.
