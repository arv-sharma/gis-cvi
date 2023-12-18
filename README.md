# Calculating community vulnerability index for flooding events through satellite image analysis
The workflow presented here is a tool to estimate community vulnerability to flooding, predicted by integrating satellite and demographic data. The extent of flooding and related metrics are calculated using Sentinel-1 and Sentinel-2 images, since these data are typically ingested and available in near real-time. Publicly available datasets on demography, economy, and terrain are then integrated in a GIS-based method. Factor analysis is used to assign weights for the different components contributing to vulnerability. The focus of this work is to provide cost-effective and timely tools to researchers and local policy-makers, and therefore, all the work was developed using open-source tools such as Python and QGIS. The methodology for calculating vulnerability index is detailed in this publication: https://doi.org/10.1016/j.rsase.2023.101086

# List of software used in this work
This tutorial utilizes two major open-source tools for analysis: Python and QGIS. The version of Python used in this work was v3.9 and was downloaded through Anaconda. QGIS v3.22 was used to visualize rasters and create training data. Numerous Python modules were downloaded to perform machine learning classifications, raster operations, and factor analysis. The list of the modules and software is provided below:

•	Python (version 3.9) downloaded from Anaconda’s website (https://docs.anaconda.com/anaconda/install/index.html).
•	QGIS (version 3.22) downloaded from https://www.qgis.org/en/site/forusers/download.html. 
•	Google Earth Engine (GEE) Python API package (‘ee’) (reference: https://developers.google.com/earth-engine/guides/python_install-conda). 
•	Rasterio package for Python.
•	Spyder IDE for Python.
•	Folium for Python.
•	Earthpy for Python.
•	XGBoost for Python.
•	Scikit-learn for Python.
•	Geopandas for Python.
•	rioxarray for Python.
•	xarray for Python.
•	osmnx for Python.
