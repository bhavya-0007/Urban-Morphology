"""Follow-up diagnostic for CHECK 2 -- run this after debug_ghsl.py confirmed
GLC_FCS30D only returns 1 image per AOI (a multi-band tile, not one image
per year). This inspects that single image's band names and properties so
we can figure out how to select the correct year's band.

Run from the project root:

    python debug_glcfcs_bands.py
"""
from __future__ import annotations

from config.settings import get_settings
from core.ee_manager import get_ee_manager
from config.datasets import GLC_FCS30D_SPEC

import ee

settings = get_settings()
ee_manager = get_ee_manager(settings)
ee_manager.ensure_initialized()
aoi = ee_manager.get_aoi_geometry()

collection = ee.ImageCollection(GLC_FCS30D_SPEC.collection_id).filterBounds(aoi)
image = collection.first()

print("=" * 70)
print("Band names on the single GLC_FCS30D tile image:")
print("=" * 70)
band_names = image.bandNames().getInfo()
for b in band_names:
    print(" ", b)
print(f"({len(band_names)} bands total)")
print()

print("=" * 70)
print("Image-level properties (look for year lists / band-to-year mapping):")
print("=" * 70)
props = image.toDictionary().getInfo()
for k, v in props.items():
    # Truncate huge values so this stays readable
    v_str = str(v)
    if len(v_str) > 200:
        v_str = v_str[:200] + " ...(truncated)"
    print(f"  {k}: {v_str}")

print()
print("Once you paste this back, we can figure out either:")
print("  (a) band names directly encode the year (e.g. 'b_2020', '2020', 'y2020') -> simple .select(bandname)")
print("  (b) bands are positional (b1..bN) and a property lists the year order -> index math needed")
