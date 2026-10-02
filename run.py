#!/usr/bin/env python3
"""Single CLI entry point for the urban morphology pipeline.

Usage:
    python run.py acquire --year 2020
    python run.py preprocess --year 2020
    python run.py indices --year 2020
    python run.py morphology --year 2020
    python run.py fishnet --year 2020
    python run.py train
    python run.py visualize
"""
from __future__ import annotations

import json
import sys
import geopandas as gpd
import geemap
from pathlib import Path

import click

from acquisition.ghsl_loader import GHSLLoader
from acquisition.glcfcs_loader import GLCFCSLoader
from acquisition.landsat_loader import LandsatLoader
from acquisition.sentinel1_loader import Sentinel1Loader
from acquisition.sentinel2_loader import Sentinel2Loader
from config.settings import get_settings
from core.ee_manager import get_ee_manager
from core.exceptions import UrbanMorphologyError
from core.exports import ExportManager
from core.logger import get_logger
from core.metadata import OutputMetadata, write_metadata
from indices import ALL_INDICES
from ml.dataset import DatasetBuilder
from ml.evaluation import ModelEvaluator
from ml.shap_analysis import SHAPAnalyzer
from ml.xgboost_model import XGBoostMorphologyClassifier
from morphology.building_density import BuildingDensityCalculator
from morphology.typology import MorphologyTypologyClassifier
from morphology.urban_extent import UrbanExtentExtractor
from fishnet.aggregator import FishnetAggregator
from fishnet.grid_generator import FishnetGenerator
from fishnet.exporter import FishnetExporter
from morphology.texture import GLCMTextureAnalyzer

logger = get_logger("run")


def _init_pipeline():
    """Build the shared Settings + EarthEngineManager for a CLI invocation."""
    settings = get_settings()
    logger_with_file = get_logger("run", level=settings.log_level, log_dir=settings.log_dir)
    ee_manager = get_ee_manager(settings)
    return settings, ee_manager, logger_with_file


def _fetch_fc_properties_batched(fc, batch_size: int = 2000, log=None):
    """Pull an already-lightweight ee.FeatureCollection's PROPERTIES (not
    geometry) into a pandas DataFrame, fetched in batches.

    CORRECTION to this function's original docstring: batching the OUTPUT
    of a FeatureCollection this way only helps when the collection itself
    is cheap to re-evaluate (e.g. a small, server-side-generated
    collection). It does NOT help when the collection was built from a
    large literal upload (e.g. geemap.geopandas_to_ee() on a big
    GeoDataFrame) or from a heavy reduceRegions() over many features —
    slicing the output afterward still re-embeds and re-evaluates that
    same expensive upstream expression on every single batch request, so
    the payload-limit problem comes right back. For those cases, chunk
    the INPUT before the expensive step instead — see
    `_aggregate_ee_chunked` below, which is what `aggregate()` actually
    uses now.

    Args:
        fc: ee.FeatureCollection to fetch properties from. Should be
            cheap to evaluate (e.g. not built from a large literal upload
            or an expensive reduceRegions()).
        batch_size: Features per batch.
        log: Optional logger for progress messages.

    Returns:
        A pandas.DataFrame with one row per feature, columns = properties.
    """
    import ee
    import pandas as pd

    n = fc.size().getInfo()
    fc_list = fc.toList(n)
    records = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = ee.FeatureCollection(fc_list.slice(start, end))
        info = batch.getInfo()
        for feat in info["features"]:
            records.append(feat["properties"])
        if log:
            log.info("Fetched feature properties %d-%d / %d", start, end, n)
    return pd.DataFrame.from_records(records)


def _aggregate_ee_chunked(gdf_wgs84, image, reducer, scale: int, batch_size: int = 1000, log=None):
    """Upload a local GeoDataFrame to Earth Engine and run reduceRegions()
    in CHUNKS, converting + reducing + fetching one small batch of LOCAL
    features at a time.

    WHY THIS IS NECESSARY (and why simpler fixes didn't work): converting
    an entire local GeoDataFrame to an ee.FeatureCollection via
    geemap.geopandas_to_ee() embeds ALL of its polygons as one literal
    constant directly in the Earth Engine expression graph — unlike a
    server-side-generated grid (e.g. aoi.coveringGrid()), which uploads
    almost nothing from the client. Earth Engine re-embeds that entire
    literal in every subsequent request that touches anything built from
    it, including something as small as `.size().getInfo()` — so neither
    batching the output afterward, nor removing an unrelated ee.Join, ever
    addressed the actual bottleneck: the one-shot upload of ~26k explicit
    polygons. Converting and reducing only ~1000 features at a time keeps
    every single request's literal payload bounded, regardless of how
    large the full grid is.

    Args:
        gdf_wgs84: The LOCAL fishnet GeoDataFrame, already reprojected to
            EPSG:4326 (required for geopandas_to_ee). Must include
            'cell_id'.
        image: The ee.Image to reduce (e.g. the multi-band `continuous`
            stack, or `typology.toInt()`).
        reducer: ee.Reducer to apply (e.g. ee.Reducer.mean() or
            ee.Reducer.mode()).
        scale: Pixel scale in meters for the reduction.
        batch_size: Local features converted+reduced per chunk. 1000 is a
            conservative starting point; lower it further if a single
            chunk still exceeds the payload limit (e.g. if more texture
            bands get added later and each feature's result grows).
        log: Optional logger for progress messages.

    Returns:
        A pandas.DataFrame with one row per feature, columns = reduced
        band values + 'cell_id' (and any other properties already on
        gdf_wgs84).
    """
    import geemap
    import pandas as pd

    n = len(gdf_wgs84)
    records = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_gdf = gdf_wgs84.iloc[start:end]
        chunk_fc = geemap.geopandas_to_ee(chunk_gdf)
        reduced = image.reduceRegions(collection=chunk_fc, reducer=reducer, scale=scale)
        info = reduced.getInfo()
        for feat in info["features"]:
            records.append(feat["properties"])
        if log:
            log.info("Aggregated + fetched cells %d-%d / %d", start, end, n)
    return pd.DataFrame.from_records(records)


@click.group()
def cli() -> None:
    """Urban Morphology Classification and Characterization of Hyderabad."""


@cli.command()
@click.option("--year", type=int, required=True, help="Study year (e.g. 2000, 2005, ... 2025).")
@click.option("--wait/--no-wait", default=False, help="Block until export tasks complete.")
def acquire(year: int, wait: bool) -> None:
    """Acquire and composite imagery for a study year (Landsat/S2/S1/GHSL/GLC_FCS30D)."""
    settings, ee_manager, log = _init_pipeline()
    if year not in settings.study_years:
        log.warning("Year %d is not one of the canonical study years %s.", year,settings.study_years,)

    aoi = ee_manager.get_aoi_geometry()
    export_manager = ExportManager(settings=settings)

    try:
        landsat_result = LandsatLoader(settings, ee_manager).load(year, aoi)
        export_manager.export_image(
            landsat_result.image, f"landsat_composite_{year}", aoi,
            scale=30, crs=settings.crs, folder_parts=("acquisition", str(year)), wait=wait,
        )

        if year >= 2017:
            s2_result = Sentinel2Loader(settings, ee_manager).load(year, aoi)
            export_manager.export_image(
                s2_result.image, f"sentinel2_composite_{year}", aoi,
                scale=10, crs=settings.crs, folder_parts=("acquisition", str(year)), wait=wait,
            )

        if year >= 2015:
            s1_result = Sentinel1Loader(settings, ee_manager).load(year, aoi)
            export_manager.export_image(
                s1_result.image, f"sentinel1_composite_{year}", aoi,
                scale=10, crs=settings.crs, folder_parts=("acquisition", str(year)), wait=wait,
            )

        ghsl_image = GHSLLoader(settings, ee_manager).load(year, aoi)
        export_manager.export_image(
            ghsl_image, f"ghsl_built_{year}", aoi,
            scale=10, crs=settings.crs, folder_parts=("acquisition", str(year)), wait=wait,
        )

        glcfcs_image = GLCFCSLoader(settings, ee_manager).load(year, aoi)
        export_manager.export_image(
            glcfcs_image, f"glcfcs_impervious_{year}", aoi,
            scale=30, crs=settings.crs, folder_parts=("acquisition", str(year)), wait=wait,
        )

        log.info("Acquisition complete for year %d.", year)
    except UrbanMorphologyError as exc:
        log.error("Acquisition failed: %s", exc)
        sys.exit(1)


@cli.command()
@click.option("--year", type=int, required=True)
def preprocess(year: int) -> None:
    """Run cloud-masking QA and generate RGB preview thumbnails for a year."""
    from preprocessing.preview import generate_rgb_thumbnail_url, save_thumbnail
    from preprocessing.quality import run_quality_checks

    settings, ee_manager, log = _init_pipeline()
    aoi = ee_manager.get_aoi_geometry()

    landsat_result = LandsatLoader(settings, ee_manager).load(year, aoi)
    report = run_quality_checks(
        landsat_result.image, aoi, scale=30, expected_ranges={"red": (0, 0.4), "nir": (0, 0.6)}
    )
    log.info("QA report: passed=%s, valid_fraction=%.2f", report.passed, report.valid_pixel_fraction)

    url = generate_rgb_thumbnail_url(landsat_result.image, aoi)
    out_dir = settings.output_dir("preprocessing", str(year))
    save_thumbnail(url, out_dir / f"landsat_preview_{year}.png")


@cli.command()
@click.option("--year", type=int, required=True)
@click.option("--wait/--no-wait", default=False)
def indices(year: int, wait: bool) -> None:
    """Compute all spectral indices for a study year and export them."""
    settings, ee_manager, log = _init_pipeline()
    aoi = ee_manager.get_aoi_geometry()
    export_manager = ExportManager(settings=settings)

    landsat_result = LandsatLoader(settings, ee_manager).load(year, aoi)
    image = landsat_result.image

    stacked = None
    for name, index_cls in ALL_INDICES.items():
        band = index_cls().compute(image)
        
        if stacked is None:
            stacked = band
        else:
            stacked = stacked.addBands(band)
            
            log.info("Computed index: %s", name)

    export_manager.export_image(
        stacked, f"spectral_indices_{year}", aoi,
        scale=30, crs=settings.crs, folder_parts=("indices", str(year)), wait=wait,
    )


@cli.command()
@click.option("--year", type=int, required=True)
@click.option("--wait/--no-wait", default=False)
def morphology(year: int, wait: bool) -> None:
    """Extract urban extent, density, and morphological typology for a year."""
    settings, ee_manager, log = _init_pipeline()
    aoi = ee_manager.get_aoi_geometry()
    export_manager = ExportManager(settings=settings)

    landsat_result = LandsatLoader(settings, ee_manager).load(year, aoi)
    image = landsat_result.image

    ndvi = ALL_INDICES["NDVI"]().compute(image)
    ndbi = ALL_INDICES["NDBI"]().compute(image)
    mndwi = ALL_INDICES["MNDWI"]().compute(image)
    bsi = ALL_INDICES["BSI"]().compute(image)
    isf = ALL_INDICES["ISF"]().compute(image)

    ghsl_image = GHSLLoader(settings, ee_manager).load(year, aoi)
    glcfcs_image = GLCFCSLoader(settings, ee_manager).load(year, aoi)

    urban_extent = UrbanExtentExtractor().extract(isf, ghsl_image, glcfcs_image, mndwi)
    density = BuildingDensityCalculator().compute_from_isf(isf)
    typology = MorphologyTypologyClassifier().classify(density, ndvi, mndwi, bsi)

    stacked = (
        urban_extent.toFloat()
        .addBands(density.toFloat())
        .addBands(typology.toFloat())
        )
    export_manager.export_image(
        stacked, f"morphology_{year}", aoi,
        scale=30, crs=settings.crs, folder_parts=("morphology", str(year)), wait=wait,
    )
    log.info("Morphology extraction complete for %d.", year)


@cli.command()
@click.option("--year", type=int, required=True)
@click.option("--cell-size", type=int, default=None, help="Override fishnet cell size (meters).")
def fishnet(year: int, cell_size: int | None) -> None:
    """Generate the fishnet grid and export it (aggregation of raster layers
    onto the grid is performed after rasters have been downloaded locally;
    see docs/IMPLEMENTATION_GUIDE.md for the full local-aggregation flow).
    """
    import geopandas as gpd
    from shapely.geometry import box

    settings, ee_manager, log = _init_pipeline()
    city = settings.city

    0
    cell = cell_size or city.fishnet_cell_size_m

    minx, miny, maxx, maxy = city.bbox_wgs84
    aoi_gdf = gpd.GeoDataFrame({"geometry": [box(minx, miny, maxx, maxy)]}, crs="EPSG:4326")

    generator = FishnetGenerator(cell_size_m=cell)
    fishnet_gdf = generator.generate(aoi_gdf, settings.crs)
    fishnet_gdf["year"] = year

    out_dir = settings.output_dir("fishnet", str(year))
    exporter = FishnetExporter()
    written = exporter.export(
        fishnet_gdf, out_dir, f"fishnet_grid_{year}", year, city.name, settings.crs, formats=("gpkg", "geojson")
    )
    log.info("Fishnet grid written: %s", list(written.values()))

@cli.command()
@click.option("--year", type=int, required=True)
def aggregate(year: int):
    """Aggregate spectral/texture/morphology layers onto the fishnet grid
    and export the combined per-cell feature table.

    IMPORTANT: this uses ONE canonical fishnet grid throughout — the local,
    precise-geometry grid written by the `fishnet` command
    (fishnet_grid_{year}.gpkg). It's uploaded to Earth Engine and reduced
    IN CHUNKS (see _aggregate_ee_chunked), then the results are merged
    straight back onto that same GeoDataFrame's geometry by cell_id.

    Do NOT swap this for a second, independently-generated EE-side grid
    (e.g. FishnetGenerator.generate_ee(), which builds cells via
    aoi.coveringGrid() and assigns cell_id purely from GEE's internal
    iteration order over grid.toList()). That order has no defined
    relationship to this local grid's cell_id (built via
    `box(x,y) for x in xs for y in ys` + AOI-intersection filtering in
    GeoPandas) — merging attribute values keyed by one grid's cell_id onto
    geometries keyed by the other's silently pairs each cell with a
    DIFFERENT physical cell's aggregated value. This previously produced a
    dataset where every exported row's indices/texture/density values
    belonged to the wrong polygon, showing up as a vertical-stripe
    artifact when visualized. Always aggregate against a FeatureCollection
    derived FROM the exact GeoDataFrame you intend to export geometry from.

    Also do NOT upload the local grid to Earth Engine in one shot (e.g.
    geemap.geopandas_to_ee() on the whole ~26k-feature GeoDataFrame at
    once) — that embeds the entire grid as a single literal constant that
    gets re-transmitted on every subsequent request built from it, which
    blows past Earth Engine's ~10MB payload limit almost immediately. Use
    _aggregate_ee_chunked, which converts and reduces a few thousand local
    features at a time.
    """
    import ee
    import geemap

    settings, ee_manager, log = _init_pipeline()
    aoi = ee_manager.get_aoi_geometry()

    # ---- ONE canonical grid, used for both aggregation and export ----
    master_fishnet_year = 2020
    fishnet_path = (
    settings.output_dir("fishnet", str(master_fishnet_year))
    / f"fishnet_grid_{master_fishnet_year}.gpkg"
    )
    fishnet_gdf = gpd.read_file(fishnet_path)
    fishnet_gdf["cell_id"] = fishnet_gdf["cell_id"].astype(str)

    # EE FeatureCollections need geographic (WGS84) geometry; reproject a
    # COPY for the EE conversion only. The original `fishnet_gdf` (in
    # settings.crs, e.g. EPSG:32644) is kept untouched and is what the
    # final export's geometry comes from, preserving exact metric cell
    # areas (cell_area_m2 stays a clean 90000.0 for 300m cells, not a
    # reprojection-noised value).-
    #
    # NOTE: deliberately NOT calling geemap.geopandas_to_ee() on the WHOLE
    # GeoDataFrame here. Doing that in one shot embeds all ~26k polygons as
    # a single literal FeatureCollection, which blows past Earth Engine's
    # payload limit on every subsequent request built from it (see
    # _aggregate_ee_chunked's docstring for the full explanation). The
    # conversion happens in bounded chunks instead, inside that helper,
    # further down.
    fishnet_gdf_wgs84 = fishnet_gdf.to_crs(4326)

    landsat_result = LandsatLoader(settings, ee_manager).load(year, aoi)
    image = landsat_result.image

    ndvi = ALL_INDICES["NDVI"]().compute(image)
    ndbi = ALL_INDICES["NDBI"]().compute(image)
    mndwi = ALL_INDICES["MNDWI"]().compute(image)
    bsi = ALL_INDICES["BSI"]().compute(image)
    ui = ALL_INDICES["UI"]().compute(image)
    isf = ALL_INDICES["ISF"]().compute(image)
    ghsl = GHSLLoader(settings, ee_manager).load(year, aoi)
    glcfcs = GLCFCSLoader(settings, ee_manager).load(year, aoi)

    texture = GLCMTextureAnalyzer(size=4)

    # -------------------------------------------------------
    # NDBI Texture
    # -------------------------------------------------------

    ndbi_texture = texture.compute_ee(
        image.addBands(ndbi),
        "NDBI",
    ).select(
        [
            "NDBI_contrast",
            "NDBI_entropy",
            "NDBI_homogeneity",
            "NDBI_correlation",
            "NDBI_energy",
        ],
        [
            "ndbi_contrast",
            "ndbi_entropy",
            "ndbi_homogeneity",
            "ndbi_correlation",
            "ndbi_energy",
        ],
    )

    # -------------------------------------------------------
    # NDVI Texture
    # -------------------------------------------------------

    ndvi_texture = texture.compute_ee(
        image.addBands(ndvi),
        "NDVI",
    ).select(
        [
            "NDVI_contrast",
            "NDVI_entropy",
            "NDVI_homogeneity",
            "NDVI_correlation",
            "NDVI_energy",
        ],
        [
            "ndvi_contrast",
            "ndvi_entropy",
            "ndvi_homogeneity",
            "ndvi_correlation",
            "ndvi_energy",
        ],
    )

    # -------------------------------------------------------
    # BSI Texture
    # -------------------------------------------------------

    bsi_texture = texture.compute_ee(
        image.addBands(bsi),
        "BSI",
    ).select(
        [
            "BSI_contrast",
            "BSI_entropy",
            "BSI_homogeneity",
            "BSI_correlation",
            "BSI_energy",
        ],
        [
            "bsi_contrast",
            "bsi_entropy",
            "bsi_homogeneity",
            "bsi_correlation",
            "bsi_energy",
        ],
    )

    # -------------------------------------------------------
    # Merge all texture layers
    # -------------------------------------------------------

    textures = (
        ndbi_texture
        .addBands(ndvi_texture)
        .addBands(bsi_texture)
    )

    urban = UrbanExtentExtractor().extract(
        isf,
        ghsl,
        glcfcs,
        mndwi,
    )

    density = BuildingDensityCalculator().compute_from_urban_extent(urban)

    typology = MorphologyTypologyClassifier().classify(
        density,
        ndvi,
        mndwi,
        bsi,
    )

    continuous = (
        ndvi
        .addBands(ndbi)
        .addBands(mndwi)
        .addBands(bsi)
        .addBands(ui)
        .addBands(isf)
        .addBands(urban)
        .addBands(density)
        .addBands(textures)
    ).toFloat()

    # NOTE: this is where the actual fix lives now. Earlier attempts
    # (removing the ee.Join, batching the OUTPUT of a single big
    # reduceRegions() call) didn't work because neither addressed the real
    # cause: fishnet_gdf_wgs84 was being uploaded to Earth Engine as ONE
    # giant literal FeatureCollection (~26k explicit polygons), which then
    # gets re-embedded in every request built from it. _aggregate_ee_chunked
    # converts + reduces + fetches ~1000 local features at a time instead,
    # so no single request ever carries more than a small slice of the
    # grid — see its docstring for the full explanation.
    log.info("Aggregating continuous (mean-reduced) bands in chunks...")
    continuous_df = _aggregate_ee_chunked(
        fishnet_gdf_wgs84, continuous, ee.Reducer.mean(), scale=30, batch_size=1000, log=log
    )
    log.info("Aggregating morphology class (mode-reduced) in chunks...")
    class_df = _aggregate_ee_chunked(
        fishnet_gdf_wgs84, typology.toInt(), ee.Reducer.mode(), scale=30, batch_size=1000, log=log
    )

    class_df = class_df.rename(columns={"mode": "morphology_class"})
    # Keep only cell_id + the class column from class_df — continuous_df
    # already carries every other property, so this avoids accidental
    # duplicate/renamed columns from a wide outer merge.
    class_df = class_df[["cell_id", "morphology_class"]]

    continuous_df["cell_id"] = continuous_df["cell_id"].astype(str)
    class_df["cell_id"] = class_df["cell_id"].astype(str)
    attr_df = continuous_df.merge(class_df, on="cell_id", how="left")

    # Merge attributes onto the SAME grid's precise local geometry.
    gdf = fishnet_gdf.merge(
        attr_df,
        on="cell_id",
        how="left"
    )

    log.info("Aggregated %d fishnet cells.", len(gdf))

    out_dir = settings.output_dir("fishnet", str(year))

    FishnetExporter().export(
        fishnet_gdf=gdf,
        output_dir=out_dir,
        base_name=f"fishnet_features_{year}",
        year=year,
        city=settings.city.name,
        crs=settings.crs,
    )

    log.info("Fishnet feature table exported successfully.")


@cli.command()
@click.option("--fishnet-csv", type=click.Path(exists=True), required=True, help="Path to an aggregated fishnet CSV with feature + target columns.")
@click.option("--target-column", default="morphology_class")
@click.option("--tune/--no-tune", default=False)
def train(fishnet_csv: str, target_column: str, tune: bool) -> None:
    """Train the XGBoost morphology classifier, evaluate it, and run SHAP."""
    import pandas as pd

    settings, ee_manager, log = _init_pipeline()

    df = pd.read_csv(fishnet_csv)
    dataset = DatasetBuilder(target_column=target_column).build(df)

    trained = XGBoostMorphologyClassifier().train(dataset, tune=tune)

    ml_dir = settings.output_dir("ml")
    XGBoostMorphologyClassifier().save(trained, ml_dir / "xgboost_morphology_model.json")

    evaluator = ModelEvaluator()
    report = evaluator.evaluate(trained)
    evaluator.save_report(report, ml_dir / "evaluation_report.json")
    log.info("Model accuracy: %.3f | F1 macro: %.3f", report.accuracy, report.f1_macro)

    shap_analyzer = SHAPAnalyzer()
    shap_result = shap_analyzer.explain(trained)
    shap_analyzer.save_summary_plot(shap_result, trained, ml_dir / "shap_summary.png")
    shap_analyzer.save_global_importance_json(shap_result, ml_dir / "shap_global_importance.json")

    write_metadata(
        ml_dir / "xgboost_morphology_model.json",
        OutputMetadata(
            product="xgboost_morphology_model",
            module="ml.xgboost_model.XGBoostMorphologyClassifier",
            parameters={"target_column": target_column, "tuned": tune, **trained.best_params},
        ),
    )
    log.info("Training pipeline complete. Outputs in %s", ml_dir)


@cli.command()
@click.option("--fishnet-csv", type=click.Path(exists=True), required=True)
@click.option("--value-column", default="NDVI")
@click.option("--year", type=int, default=None)
def visualize(fishnet_csv: str, value_column: str, year: int | None) -> None:
    """Build static and HTML report figures from an aggregated fishnet CSV."""
    import geopandas as gpd
    import pandas as pd

    settings, ee_manager, log = _init_pipeline()
    df = pd.read_csv(fishnet_csv)

    df = df.dropna().reset_index(drop=True)

    from shapely import wkt

    if "geometry" in df.columns and df["geometry"].dtype == object:
        df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=settings.crs)

    viz_dir = settings.output_dir("visualization", str(year) if year else "latest")

    from visualization.html_export import HTMLReportBuilder
    from visualization.maps import InteractiveMapBuilder

    map_builder = InteractiveMapBuilder()
    map_path = map_builder.build_vector_map(gdf, value_column, viz_dir / f"{value_column}_map.png", title=value_column)

    report = HTMLReportBuilder(title=f"Urban Morphology Report — {settings.city.name}")
    report.add_image_section(f"{value_column} distribution", map_path, caption=f"Fishnet cell values of {value_column}")
    report_path = report.build(viz_dir / "report.html")

    log.info("Visualization outputs written to %s", viz_dir)
    log.info("Report: %s", report_path)


if __name__ == "__main__":
    cli()