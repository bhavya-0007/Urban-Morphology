# Urban Morphology Classification and Characterization — Hyderabad (2000–2025)

A research-grade, modular Python pipeline for multi-source remote-sensing
urban morphology analysis, built around Google Earth Engine, spectral
indices, texture/landscape metrics, Local Climate Zone (LCZ) and
morphological typology classification, a 300 m fishnet spatial framework,
and explainable machine learning (XGBoost + SHAP).

This implementation follows the accompanying **Project Design Document
(PDD)** exactly: layered architecture, one class per spectral index,
configuration-driven behavior, full type hints, logging (no `print`),
and unit-testable components.

## Quick start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env: set EE_PROJECT to your Earth Engine cloud project

# 4. Authenticate Earth Engine (first run only — opens a browser)
earthengine authenticate

# 5. Run the test suite
pytest -q

# 6. Run pipeline stages
python run.py acquire --year 2020
python run.py preprocess --year 2020
python run.py indices --year 2020
python run.py morphology --year 2020
python run.py fishnet --year 2020
python run.py train --fishnet-csv outputs/fishnet/2020/fishnet_features_2020.csv
python run.py visualize --fishnet-csv outputs/fishnet/2020/fishnet_features_2020.csv
```

See **`docs/IMPLEMENTATION_GUIDE.md`** for the full end-to-end workflow
(including the local raster-to-fishnet aggregation step that bridges
`morphology` and `train`), and **`docs/ARCHITECTURE.md`** for the module
map and design rationale.

## Project structure

```
urban_morphology/
  config/          Settings, dataset registry, constants, city definitions
  core/            Logging, EE session mgmt, exports, validation, metadata, exceptions, utils
  acquisition/     Landsat / Sentinel-2 / Sentinel-1 / GHSL / GLC_FCS30D loaders
  preprocessing/   Cloud masking helpers, compositing, clipping, previews, QA
  indices/         NDVI, NDBI, MNDWI, IBI, BSI, UI, ISF — one class each
  morphology/      Urban extent, building density/spacing, texture, landscape metrics, LCZ, typology
  fishnet/         300 m grid generation, aggregation, export
  ml/              Dataset prep, XGBoost, evaluation, SHAP explainability
  visualization/   Interactive maps, static plots, HTML report bundling
  tests/           43 unit tests (config, indices, utils, validation)
  outputs/         All generated artifacts (gitignored except .gitkeep)
  docs/            Implementation guide + architecture notes
  run.py           Single CLI entry point (Click)
  requirements.txt
```

## Study parameters

| Parameter | Value |
|---|---|
| City | Hyderabad, Telangana, India |
| Projection | EPSG:32644 (UTM 44N) |
| Study years | 2000, 2005, 2010, 2015, 2020, 2025 |
| Season | Pre-monsoon (Feb–May) |
| Fishnet resolution | 300 m |
| Scope | Urban morphology only — SUHI/LST/thermal explicitly excluded |

## Data sources

Landsat TM/ETM+/OLI (C02 L2), Sentinel-2 L2A, Sentinel-1 GRD, JRC GHSL
built-up surface, GLC_FCS30D land cover, and OpenStreetMap building
footprints/street networks (for spacing metrics).

## Notes on required Earth Engine access

This pipeline calls the Earth Engine Python API directly (`ee.Image`,
`ee.ImageCollection`, `ee.batch.Export`, ...). You need:

1. A Google account registered for Earth Engine access
   (https://code.earthengine.google.com/register).
2. A Google Cloud project with the Earth Engine API enabled, referenced
   by `EE_PROJECT` in `.env`.
3. For unattended/CI runs, a service account key
   (`EE_SERVICE_ACCOUNT` + `EE_CREDENTIALS_PATH`) instead of interactive
   `earthengine authenticate`.

## License / attribution

This is a research tool. Cite the underlying data providers (USGS/NASA
Landsat, ESA Copernicus Sentinel-1/2, JRC GHSL, GLC_FCS30D authors, and
OpenStreetMap contributors) in any resulting publication.
