"""One-off diagnostic script — checks two open questions from the GHSL/GLC_FCS30D review:

  1. Is GHS_BUILT_S actually a [0,1] fraction, or m^2 built-up area / percent?
  2. Does GLC_FCS30D's system:index look like a bare year ("2020") or
     something longer (e.g. "GLC_FCS30D_20200101-20201231")?

Run from the project root (same directory as run.py) so the existing
package imports resolve:

    python debug_ghsl.py

No changes to your pipeline code are needed for this -- it's read-only.
"""
from __future__ import annotations

from config.settings import get_settings
from core.ee_manager import get_ee_manager
from config.datasets import GHSL_BUILT_SPEC, GLC_FCS30D_SPEC

import ee

settings = get_settings()
ee_manager = get_ee_manager(settings)
ee_manager.ensure_initialized()

aoi = ee_manager.get_aoi_geometry()

# ---------------------------------------------------------------------------
# Check 1: GHSL built-surface value range/units
# ---------------------------------------------------------------------------
print("=" * 70)
print("CHECK 1: GHSL built-surface band -- what range does it actually hold?")
print("=" * 70)

ghsl_collection = ee.ImageCollection(GHSL_BUILT_SPEC.collection_id)

# Same epoch-matching logic as GHSLLoader, done manually here for year 2020.
epoch = 2020
ghsl_image = ghsl_collection.filter(ee.Filter.eq("system:index", str(epoch))).first()
ghsl_image = ee.Image(ghsl_image).select([0])

stats = ghsl_image.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=aoi,
    scale=100,
    bestEffort=True,
    maxPixels=1e13,
).getInfo()

print("GHSL band min/max over the AOI:", stats)
print()
print("Interpretation guide:")
print("  - values roughly within [0, 1]      -> already a fraction, ghsl_threshold=0.2 is fine as-is")
print("  - values roughly within [0, 100]    -> it's a percent, threshold needs to be e.g. 20, not 0.2")
print("  - values roughly within [0, 10000]  -> it's m^2 built-up area per 100m pixel, needs /10000 or a different threshold entirely")
print()

# ---------------------------------------------------------------------------
# Check 2: GLC_FCS30D system:index format
# ---------------------------------------------------------------------------
print("=" * 70)
print("CHECK 2: GLC_FCS30D -- what do the actual system:index strings look like?")
print("=" * 70)

glcfcs_collection = ee.ImageCollection(GLC_FCS30D_SPEC.collection_id).filterBounds(aoi)
index_list = glcfcs_collection.aggregate_array("system:index").getInfo()

print("Distinct system:index values found (first 20 shown):")
for idx in index_list[:20]:
    print(" ", idx)
print(f"... {len(index_list)} total")
print()
print("Interpretation guide:")
print("  - if these are bare years like '2020'                -> glcfcs_loader.py's filter is correct as-is")
print("  - if these look like 'GLC_FCS30D_20200101-20201231'  -> ee.Filter.eq('system:index', str(epoch)) never")
print("    matches, .first() silently returns an empty image, and 'impervious' will be all-zero -- needs a")
print("    ee.Filter.stringContains('system:index', str(epoch)) or similar instead of an exact match.")
