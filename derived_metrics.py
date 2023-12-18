# -*- coding: utf-8 -*-
"""
Created on Mon Jul 4 20:45:12 2022

This script processes various rasters and performs different calculations on them to generate derived rasters.

"""

import rioxarray as rxr
import os
import glob
import numpy as np
from scipy.ndimage import generic_filter
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

#%% Function to process folders

def processRasters(rootFolder, folder):
    """
    Process rasters and generate derived rasters for a given location.

    Parameters
    ----------
    rootFolder : str
        Root folder containing the reprojected data folders.
    folder : str
        Sub-folder identifying the location.

    Returns
    -------
    None.

    """
    targetFolder = os.path.join(rootFolder, folder, 'derived')
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)

    #%% GDP 2015
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'GDP2015.tif')) as GDP2015:
        GDP2015_data = GDP2015[0].where(GDP2015[0] >= 0, other=np.nan)
    # GDP2015_data.rio.to_raster(os.path.join(targetFolder, 'GDP2015.tif'))

    #%% LULC Human modification
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'lulc_human_mod.tif')) as LULCHumanMod:
        LULCHumanMod_data = LULCHumanMod[0].where(LULCHumanMod[0] >=0, other=np.nan)
    LULCHumanMod_data.rio.to_raster(os.path.join(targetFolder, 'LULC_HumanMod.tif'))

    #%% POPDens2020
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'popDens_2020.tif')) as popDens2020:
        popDens2020_data = popDens2020[0].where(popDens2020[0] >=0, other=np.nan)
    # popDens2020_data.rio.to_raster(os.path.join(targetFolder, 'popDens2020.tif'))

    #%% GDPIncrease
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'GDP2000.tif')) as GDP2000:
        GDP2000_data = GDP2000[0].where(GDP2000[0] >= 0, other=np.nan)
    GDPIncrease = GDP2015_data - GDP2000_data
    GDPIncrease.rio.to_raster(os.path.join(targetFolder, 'GDPIncrease.tif'))

    #%% PopDensIncrease
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'popDens_2000.tif')) as popDens2000:
        popDens2000_data = popDens2000[0].where(popDens2000[0] >=0, other=np.nan)
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'popDens_2015.tif')) as popDens2015:
        popDens2015_data = popDens2015[0].where(popDens2015[0] >=0, other=np.nan)
    popDensIncrease = popDens2015_data - popDens2000_data
    popDensIncrease.rio.to_raster(os.path.join(targetFolder, 'popDensIncrease.tif'))

    #%% Under 14 population density
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'popDens_a0to14.tif')) as popDens_a0to14:
        popDens_a0to14_data = popDens_a0to14[0].where(popDens_a0to14[0] >=0, other=np.nan)
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'popDens_all.tif')) as popDens_all:
        popDens_all_data = popDens_all[0].where(popDens_all[0] >=0, other=np.nan)
        popDens_all_data = popDens_all_data.where(popDens_a0to14_data != 0, other=1e10) # Set arbitrary high value to prevent divide by zero error
    popDens_a0to14_norm = np.divide(popDens_a0to14_data, popDens_all_data)
    popDens_a0to14_norm.rio.to_raster(os.path.join(targetFolder, 'popDens_a0to14_norm.tif'))

    #%% Over 65 population density
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'popDens_a65plus.tif')) as popDens_a65plus:
        popDens_a65plus_data = popDens_a65plus[0].where(popDens_a65plus[0] >=0, other=np.nan)
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'popDens_all.tif')) as popDens_all:
        popDens_all_data = popDens_all[0].where(popDens_all[0] >=0, other=np.nan)
        popDens_all_data = popDens_all_data.where(popDens_a65plus_data != 0, other=1e10) # Set arbitrary high value to prevent divide by zero error
    popDens_a65plus_norm = np.divide(popDens_a65plus_data, popDens_all_data)
    popDens_a65plus_norm.rio.to_raster(os.path.join(targetFolder, 'popDens_a65plus_norm.tif'))

    #%% Female population density
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'popDens_female.tif')) as popDens_female:
        popDens_female_data = popDens_female[0].where(popDens_female[0] >=0, other=np.nan)
    with rxr.open_rasterio(os.path.join(rootFolder, folder, 'popDens_all.tif')) as popDens_all:
        popDens_all_data = popDens_all[0].where(popDens_all[0] >=0, other=np.nan)
        popDens_all_data = popDens_all_data.where(popDens_female_data != 0, other=1e10) # Set arbitrary high value to prevent divide by zero error
    popDens_female_norm = np.divide(popDens_female_data, popDens_all_data)
    popDens_female_norm.rio.to_raster(os.path.join(targetFolder, 'popDens_female_norm.tif'))

    #%% Inundation metric
    with rxr.open_rasterio(os.path.join(r'REPLACE_WITH_DEM_PATH', 'GEE_' + folder, 'inundationMetric.tif')) as inundationMetric:
        inundationMetricData = np.sqrt(inundationMetric[0])  # Change made on 20231015 to get geometric mean
    inundationMetricData_repr = inundationMetricData.rio.reproject_match(popDens2020_data, nodata=np.nan)
    inundationMetricData_repr.rio.to_raster(os.path.join(targetFolder, 'inundationMetric_repr.tif'))

    #%% Built-area metrics (Uncomment and customize as needed)
    # BFFImgFolder = glob.glob(os.path.join(r'REPLACE_WITH_BFF_FOLDER', 'GEE_' + folder + '_S2_BFF*'))
    # with rxr.open_rasterio(os.path.join(BFFImgFolder[0], 'classified_' + folder + '_BFF.tif')) as BFFImg:
    #     BFFImgData = BFFImg[0]
    # builtUpFraction = generic_filter(BFFImgData.astype('float64'), lambda array: np.count_nonzero(array == 2), footprint=np.ones((45, 45)), mode='nearest') / 45 ** 2   # To match lookup area to pixel width in GPW raster
    # builtUpFractionRaster = BFFImgData.copy(data=builtUpFraction)
    # builtUpFractionRaster_repr = builtUpFractionRaster.rio.reproject_match(popDens2020_data, nodata=np.nan)
    # builtUpFractionRaster_repr.rio.to_raster(os.path.join(targetFolder, 'builtUpAreaFraction.tif'))

    #%% Flooded area metrics (Uncomment and customize as needed)
    # AFFImgFolder = glob.glob(os.path.join(r'REPLACE_WITH_AFF_FOLDER', 'GEE_' + folder + '_S2_AFF*'))
    # with rxr.open_rasterio(os.path.join(AFFImgFolder[0], 'landCoverChange.tif')) as compImg:
    #     compImgData = compImg[0]
    # floodPixels = generic_filter(compImgData.astype('float64'), lambda array: np.count_nonzero(np.isin(array, [5, 9, 13])), footprint=np.ones((45, 45)), mode='nearest')   # To match lookup area to pixel width in GPW raster
    # nonNanPixels = generic_filter(compImgData.astype('float64'), lambda array: np.count_nonzero(array != 251), footprint=np.ones((45, 45)), mode='nearest')
    # floodFraction = np.divide(floodPixels, nonNanPixels)
    # floodFractionRaster = compImgData.copy(data=floodFraction)
    # floodFractionRaster = floodFractionRaster.where(nonNanPixels > 0, other=np.nan)
    # floodFractionRaster_repr = floodFractionRaster.rio.reproject_match(popDens2020_data, nodata=np.nan)
    # floodFractionRaster_repr.rio.to_raster(os.path.join(targetFolder, 'floodAreaFraction.tif'))

    #%% Infrastructure and amenities metric
    # with rxr.open_rasterio(os.path.join(rootFolder, folder, 'infra_OSM.tif')) as infra_OSM:
    #     infra_OSM_data = infra_OSM[0]
    # infra_OSM_data.rio.to_raster(os.path.join(targetFolder, 'infra_OSM.tif'))

#%% Main

if __name__ == '__main__':
    rootFolder = 'REPLACE_WITH_ROOT_FOLDER_PATH'  # Replace with the actual root folder path
    # BFFFolder = 'REPLACE_WITH_BFF_ROOT_PATH'  # Uncomment and replace with BFF root folder path
    # AFFFolder = 'REPLACE_WITH_AFF_ROOT_PATH'  # Uncomment and replace with AFF root folder path

    Parallel(n_jobs=cpu_count())(delayed(processRasters)(rootFolder, folder) for folder in tqdm(os.listdir(rootFolder), total=len(os.listdir(rootFolder)), desc='Progress'))
