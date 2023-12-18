# -*- coding: utf-8 -*-
"""
Created on Thu Jul 7 19:44:00 2022

This script retrieves OpenStreetMap (OSM) data for specified infrastructure types, counts occurrences within each pixel of a reference raster, and writes the counts as a raster file.

"""

import geopandas as gpd
import numpy as np
import osmnx as ox
import rioxarray as rxr
import os
from shapely.geometry import Polygon
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

#%% Function to get OSM data and write it as a raster

def processOSMData2Raster(rootFolder, folder, rawDataFolder):
    """
    Process OSM data and write counts as a raster.

    Parameters
    ----------
    rootFolder : str
        Folder with the reprojected data folders.
    folder : str
        Sub-folder identifying the location name.
    rawDataFolder : str
        Folder to store OSM data.

    Returns
    -------
    None.

    """
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'popDens_2020.tif')) as src:
        src_data = src[0]
        
    # Generate polygons based on raster pixels
    polygons = []
    dx, dy = src.rio.resolution()[0] / 2, src.rio.resolution()[1] / 2
    for y in src.y.values:
        for x in src.x.values:
            polygons.append(Polygon([(x - dx, y - dy), (x + dx, y - dy), (x + dx, y  + dy), (x - dx, y + dy), (x - dx, y - dy)]))
    
    # Pixel grid
    pxGrid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:4326')
    
    # Define OSM data tags
    tags = {'amenity': ['fuel', 'atm', 'bank', 'clinic', 'doctors', 'hospital', 'nursing_home', 'pharmacy', 'social_facility', 'veterinary', 'community_centre', 'social_centre', 'fire_station', 'police', 'townhall', 'drinking_water', 'shelter', 'telephone', 'toilets', 'water_point'],
            'emergency': ['ambulance_station', 'landing_site', 'assembly_point', 'phone', 'siren', 'drinking_water'], 
            'healthcare': True,
            'highway': ['motorway', 'trunk', 'primary'],
            'office': ['government', 'newspaper'],
            'public_transport': 'station',
            'shop': ['convenience', 'general', 'supermarket', 'hardware']}
    
    # Get OSM data
    G = ox.geometries_from_bbox(src.rio.bounds()[3], src.rio.bounds()[1], src.rio.bounds()[2], src.rio.bounds()[0], tags)
    
    # Find intersection between pixel grid polygons and geometry of OSM data
    intersection_gdf = gpd.sjoin(pxGrid_gdf, G, how='inner', predicate='intersects')
    intersection_idx = intersection_gdf.index.to_numpy()
    
    # Count number of instances in each pixel
    counts = [np.count_nonzero(intersection_idx == idx) for idx in pxGrid_gdf.index]
    
    # Write the counts as raster
    infrRaster = src_data.copy(data=np.asarray(counts, dtype='float64').reshape(src_data.shape))
    infrRaster.rio.to_raster(os.path.join(rootFolder, folder, 'infra_OSM.tif'))
    
    # Write OSM data as a GeoDataFrame
    if not os.path.isdir(os.path.join(rawDataFolder, folder)):
        os.makedirs(os.path.join(rawDataFolder, folder))
    columns2Drop = [column for column in G.columns if any(isinstance(item, list) for item in G[column])]
    try:
        G.drop(columns=columns2Drop).to_file(os.path.join(rawDataFolder, folder, 'rawData.gpkg'), driver='GPKG')
    except RuntimeError:
        print('Runtime error occurred while saving raw data for ' + folder)
    
#%% Main

if __name__ == '__main__':
    rootFolder = 'YOUR_ROOT_FOLDER_PATH'  # Replace with the actual root folder path
    rawDataFolder = 'YOUR_RAW_DATA_FOLDER_PATH'  # Replace with the actual raw data folder path
    Parallel(n_jobs=cpu_count())(delayed(processOSMData2Raster)(rootFolder, folder, rawDataFolder) for folder in tqdm(os.listdir(rootFolder), total=len(os.listdir(rootFolder)), desc='Progress'))
