import geopandas as gpd

# Read the fishnet
gdf = gpd.read_file("outputs/fishnet/2020/fishnet_features_2020.gpkg")

print("Original CRS:", gdf.crs)

# Convert to WGS84
gdf = gdf.to_crs("EPSG:4326")

print("Converted CRS:", gdf.crs)

# Save as ESRI Shapefile
gdf.to_file(
    "outputs/fishnet/2020/fishnet_features_2020_gee.shp",
    driver="ESRI Shapefile"
)

print("Done.")