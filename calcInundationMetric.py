# -*- coding: utf-8 -*-
"""
Created on Fri Jul 1 12:34:11 2022

This script calculates inundation risk based on slope and aspect information.

"""

import numpy as np
import rioxarray as rxr
import glob
import os
from scipy.ndimage import generic_filter
from skimage.morphology import disk
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

#%% Function to calculate inundation risk at a point

def calcInundationRisk(kernel):
    """
    Calculates inundation risk in a kernel by comparing the aspect in the kernel with ideal minimum and maximum risk aspect matrices.
    Maximum risk would be 1, minimum 0.

    Parameters
    ----------
    kernel : array of integers or floats from image.
        Flattened array of values from an image acquired by sliding a window across image pixels.

    Returns
    -------
    inundationRisk : float
        A metric of inundation risk.

    """
    kernelMod = np.delete(kernel, int((kernel.size - 1) / 2))   # Delete the center pixel of the disk since aspect at center is not defined
    aspectDiff = kernelMod - inundMinRiskAspect_flat
    aspectDiffMin = np.minimum(aspectDiff % 360, 360 - (aspectDiff % 360))
    inundationRisk = np.dot(aspectDiffMin, inundMaxRiskAspect) / inundMaxRiskAspectNorm ** 2
    return inundationRisk

#%% Function to process the files given a location

def processFiles(folder, diskRadius):
    """
    Process slope and aspect files to calculate inundation risk.

    Parameters
    ----------
    folder : str
        Directory containing slope and aspect files.
    diskRadius : int
        Radius of the disk for filtering.

    Returns
    -------
    None.

    """
    with rxr.open_rasterio(os.path.join(folder, 'aspect.tif')) as aspect:
        aspectImg = aspect[0].to_numpy()
    inundationRiskfromAspect = generic_filter(aspectImg, calcInundationRisk, footprint=disk(diskRadius), mode='constant', cval=np.nan)
    print('Processed inundation risk from aspect.\n')
    with rxr.open_rasterio(os.path.join(folder, 'slope.tif')) as slope:
        inundationRiskfromSlope = (90 - slope[0].to_numpy()) / 90
    inundationRiskfromAspect_filt = generic_filter(inundationRiskfromAspect, lambda array: np.count_nonzero(array >= 0.75), footprint=np.ones((45, 45)), mode='nearest') / 45 ** 2   # To match lookup area to pixel width in GPW raster
    # inundationRiskfromAspect_filt = generic_filter(inundationRiskfromAspect, lambda array: np.count_nonzero(array >= 0.75), footprint=disk(8 * diskRadius), mode='nearest') / np.count_nonzero(disk(8 * diskRadius))
    print('Processed filtering of inundation risk from aspect.')
    totalInundationRisk = np.multiply(inundationRiskfromAspect_filt, inundationRiskfromSlope)
    inundationMetricRaster = aspect # Use aspect file's metadata to write raster
    inundationMetricRaster[0] = totalInundationRisk
    inundationMetricRaster.rio.to_raster(os.path.join(folder, 'inundationMetric.tif')) 
    print('File written to {}'.format(os.path.join(folder, 'inundationMetric.tif')))
    
#%% Run main function

if __name__ == '__main__':
    diskRadius = 3
    kernelShape = (diskRadius * 2 + 1, diskRadius * 2 + 1) # Needs to be odd and square
    kernelSize = kernelShape[0]
    x = np.arange(-(kernelSize - 1) / 2, ((kernelSize - 1) / 2 + 1), 1)
    y = x  # Since kernel is square, indices are the same in both directions
    X, Y = np.meshgrid(x, y)
    # Get a matrix representing perfectly inward facing slopes. From GEE: Units are degrees where 0=N, 90=E, 180=S, 270=W. See example in https://developers.google.com/earth-engine/apidocs/ee-terrain-slope
    inundMinRiskAspect = (np.rad2deg(np.arctan2(Y, X)) + 90) % 360
    inundMinRiskAspect_flat = inundMinRiskAspect[np.array(disk(diskRadius), dtype=bool)]
    inundMinRiskAspect_flat = np.delete(inundMinRiskAspect_flat, int((inundMinRiskAspect_flat.size - 1) / 2))
    inundMaxRiskAspect = np.ones(inundMinRiskAspect_flat.shape) * 180
    inundMaxRiskAspectNorm = np.linalg.norm(inundMaxRiskAspect)
    folders = glob.glob(os.path.join(r'YOUR_SOURCE_DIRECTORY', 'GEE*'))  # Replace with the actual source directory
    Parallel(n_jobs=cpu_count())(delayed(processFiles)(folder, diskRadius) for folder in tqdm(folders, total=len(folders), desc='Progress'))
