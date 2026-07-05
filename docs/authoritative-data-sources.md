# Authoritative Data Sources

This project should prefer authoritative meteorite references before broad web-image scraping.

## Why

Manual screening and box annotation are expensive. Authoritative databases and museum collections reduce noise by starting from confirmed meteorites, known classifications, and traceable sources.

## Primary Sources

### Meteoritical Bulletin Database

URL: https://www.lpi.usra.edu/meteor/

Use for:

- Official meteorite names
- Classifications
- Find/fall status
- Locality and mass metadata
- Filtering records with photographs
- CSV-style metadata exports where available

Role in RKHunter:

Use as the master index for confirmed meteorites and metadata. Do not treat every record as training-ready imagery.

### NASA Antarctic Meteorite Program

URL: https://curator.jsc.nasa.gov/antmet/

Use for:

- Antarctic meteorite collection context
- Curated sample references
- Scientific collection metadata

Role in RKHunter:

Use as a high-trust reference source for meteorite samples and institutional metadata.

### Smithsonian National Museum of Natural History

URL: https://collections.nmnh.si.edu/

Use for:

- Museum collection records
- Specimen metadata
- High-trust reference imagery when available

Role in RKHunter:

Use for reference images and specimen metadata. Check license and reuse terms before redistribution.

### Natural History Museum London

URL: https://data.nhm.ac.uk/

Use for:

- Collection metadata
- Museum specimen records
- Potential reference imagery and identifiers

Role in RKHunter:

Use for reference records and license-aware image collection.

## Data Tiers

### Tier 1: Confirmed Meteorite References

High-trust museum or scientific collection images. These teach the project what known meteorites look like.

Store locally under:

```text
data/external/authoritative-meteorites/
data/external/museum-images/
```

### Tier 2: Task Backgrounds

Desert, Gobi, dry lake bed, salt flat, and barren gravel imagery. These teach the model what the search terrain looks like.

### Tier 3: Mission-Like Images

Phone-simulated or drone-captured imagery with candidate rocks placed in real terrain. This is the most important data for final detection performance.

## License Rules

- Keep source URL, institution, specimen ID, license, and access date in metadata.
- Do not upload large third-party image folders to GitHub.
- Prefer storing manifests, scripts, and documentation in Git.
- Before public redistribution, verify each source's reuse policy.
