# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:33:36 2022
"""
# -*- coding: utf-8 -*-

import os
import glob
import rasterio as rio
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.interpolate import interp2d
from joblib import Parallel, delayed, cpu_count
from getFileName import getDirectory
from tqdm import tqdm

#%% Function to calculate grey level covariance matrix and metrics

def calcGLCMMetrics(array):
    """
    Receives a slice of an image, computes Gray Level Covariance Matrix (GLCM) and associated metrics.

    Parameters
    ----------
    array : Numpy array of type 'uint8'
        The slice of the image that will be used to calculate Gray Level Covariance Matrix.

    Returns
    -------
    A tuple containing lists of length corresponding to the number of image slices in input array. Each list in the tuple contains a metric calculated from GLCM.

    """
    
    if np.isnan(array).any():
        return (np.nan, np.nan, np.nan)
    
    contrast = []
    energy = []
    correlation = []
    for imgNum in range(array.shape[0]):
        Z = array[imgNum, :, :]
        GLCM = graycomatrix(Z, [1], [0], levels=24, symmetric=True, normed=True)
        contrast.append(graycoprops(GLCM, 'contrast').item())
        energy.append(graycoprops(GLCM, 'energy').item())
        correlation.append(graycoprops(GLCM, 'correlation').item())
    
    return (contrast, energy, correlation)

#%% Function to generate texture features

def genTextureFeatures(inpFileName, winSize):
    """
    This function receives a window-size for Gray-Level Covariance Matrix, and asks the user to identify the image to generate texture features and an output filename to save the results. The function calls calcGLCMMetrics() to calculate features for each of the input bands and then saves the result.

    Parameters
    ----------
    winSize : int
        An odd integer less than the size of the image.

    Returns
    -------
    None.

    """
	
    print('Processing {} ...'.format(inpFileName))

    ## Check if the window size is an odd number
    if (winSize % 2) == 0:
        raise ValueError('Window size is even. Choose an odd window size and rerun.')

    ## Output file name
    outputFileName = ''.join([inpFileName.replace('Input', 'Output').rstrip('.tif'), '_texture.tif'])
    
    ## Open image

    with rio.open(inpFileName) as s1Img:
    
        # Get profile of S1 image to use later as metadata for writing rasters
        s1ImgProfile = s1Img.profile
        
        # Get descriptions
        s1Img_descriptions = s1Img.descriptions
    
        # Create a numpy stack
        s1Img_list = []
        for bandNum in range(1, 1 + s1Img.count):
            s1Img_list.append(s1Img.read(bandNum))
            
        s1Img_array = np.stack(s1Img_list)
        s1ImgNorm_array = np.zeros(s1Img_array.shape, dtype='uint8')
        for imgNum in range(s1Img_array.shape[0]):
            s1ImgNorm_array[imgNum, :, :] = (np.interp(s1Img_array[imgNum, :, :], (np.nanmin(s1Img_array[imgNum, :, :]), np.nanmax(s1Img_array[imgNum, :, :])), (0, 23))).astype('uint8')            
    
    ## Number of valid pixels
    validPxShape = (s1Img.shape[0] - (winSize - 1), s1Img.shape[1] - (winSize - 1))
    numValidPx = validPxShape[0] * validPxShape[1]
    
    ## Get data using paralled for loop
    combinedOutput = Parallel(n_jobs=cpu_count())(delayed(calcGLCMMetrics)(s1ImgNorm_array[:, (np.floor(ii / validPxShape[1])).astype(int): (np.floor(ii / validPxShape[1]) + winSize).astype(int), (ii % validPxShape[1]): (ii % validPxShape[1]) + winSize]) for ii in tqdm(range(numValidPx), desc='Progress: '))
    
    contrastList = [item[0] for item in combinedOutput]
    energyList = [item[1] for item in combinedOutput]
    correlationList = [item[2] for item in combinedOutput]
    
    GLCM_Data = {}
    x = np.arange((winSize - 1) / 2, (winSize - 1) / 2 + validPxShape[1])
    y = np.arange((winSize - 1) / 2, (winSize - 1) / 2 + validPxShape[0])
    for (desc, imgNum) in zip(s1Img_descriptions, range(s1Img.count)):
        _temp = np.asarray([item[imgNum] for item in contrastList]).reshape(validPxShape)
        tempInterpolator = interp2d(x, y, _temp)
        tempData = tempInterpolator(np.arange(s1Img.shape[1]), np.arange(s1Img.shape[0]))
        tempData[np.isnan(s1Img_array[imgNum, :, :])] = np.nan
        GLCM_Data[desc + '_' + 'contrast'] = tempData
        _temp = np.asarray([item[imgNum] for item in energyList]).reshape(validPxShape)
        tempInterpolator = interp2d(x, y, _temp)
        tempData = tempInterpolator(np.arange(s1Img.shape[1]), np.arange(s1Img.shape[0]))
        tempData[np.isnan(s1Img_array[imgNum, :, :])] = np.nan
        GLCM_Data[desc + '_' + 'energy'] = tempData
        _temp = np.asarray([item[imgNum] for item in correlationList]).reshape(validPxShape)
        tempInterpolator = interp2d(x, y, _temp)
        tempData = tempInterpolator(np.arange(s1Img.shape[1]), np.arange(s1Img.shape[0]))
        tempData[np.isnan(s1Img_array[imgNum, :, :])] = np.nan
        GLCM_Data[desc + '_' + 'correlation'] = tempData
    
    ## Writing data to local drive
    
    with rio.Env():

        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source

        # And then change the band count to 1, set the
        # dtype to float64, and specify LZW compression.
        s1ImgProfile.update(
            dtype=rio.float64,
            count=(s1Img.count + len(GLCM_Data)),
            compress='lzw')
        
        # Write classified image as raster
        with rio.open(outputFileName, 'w', **s1ImgProfile) as dst:
            
            # First write the original bands in dB
            for bandNum in range(1, 1 + s1Img.count):
                dst.write_band(bandNum, s1Img_array[bandNum - 1, :, :])
                dst.set_band_description(bandNum, s1Img.descriptions[bandNum - 1])
            # Next write the texture bands
            for newBandNum, (bandName, bandData) in enumerate(GLCM_Data.items(), start=(bandNum + 1)):
                dst.write_band(newBandNum, bandData)
                dst.set_band_description(newBandNum, bandName)
                
    print('Image with features saved at location {} .'.format(outputFileName))
            
            
#%% Main function

def main():
    """
    Main function. Runs a while loop to process S-1 images and generate features until the user exits by pressing 0 on a prompt.

    Returns
    -------
    None.

    """

    winSize = 5
    imgFolder = getDirectory('Select the folder with S-1 images: ')
    S1Folders = glob.glob(os.path.join(imgFolder, '*_S1_*'))
    for folderName in S1Folders:
    	genTextureFeatures(glob.glob(os.path.join(folderName, '*_dB.tif'))[0], winSize)
        

#%% Run main function

if __name__ == '__main__':
    main()

