## Urban Morphology & Surface Urban Heat Island Analysis

A geospatial and machine learning project investigating how **urban morphology, land-cover characteristics, and spatial structure influence Surface Urban Heat Island (SUHI) intensity** across Indian cities.

The project combines **Landsat, MODIS, and other Earth observation datasets** with Google Earth Engine and Python-based geospatial processing to characterize urban environments and study their relationship with land surface temperature.

### What I've worked On

* Extracting **urban and rural extents** using NDBI, NDVI, BSI, and Otsu thresholding.
* Calculating spectral and built-environment indicators including **NDVI, NDBI, MNDWI, BSI, impervious surface fraction, and building density**.
* Using **GHS-BUILT-S** building data to quantify urban building density and spatial structure.
* Generating spatial **fishnet grids** to standardize and aggregate urban morphology variables.
* Developing **urban morphology typologies** based on built-up density, vegetation, imperviousness, and other spatial characteristics.
* Analyzing relationships between morphology variables and **Land Surface Temperature (LST)**.
* Estimating **SUHI intensity** using the difference between urban and rural LST.
* Applying **XGBoost and SHAP** to classify and interpret urban morphology patterns.
* Comparing spatial and seasonal patterns across multiple years and Indian cities.

### Technologies & Data

**Google Earth Engine · Python · Remote Sensing · GIS · Machine Learning · XGBoost · SHAP · Landsat · MODIS · GHS-BUILT-S · GLC_FCS30D**

The broader study examines **Delhi, Mumbai, Chennai, Hyderabad, Kolkata, Bhopal, Patna, Srinagar, Raipur, and Kochi** across multiple time periods and seasonal conditions.

### Goal

The goal is to understand how **the way cities are built and organized affects urban thermal environments**, and to develop an interpretable, data-driven framework for studying the relationship between **urban form, land cover, and heat intensity**.
