# -*- coding: utf-8 -*-
"""
Script for visualizing and analyzing land cover changes before and after a flash flood.

Created on Fri May 6 12:41:50 2022
"""

import os
import rioxarray as rxr
import rasterio.features as rioFeatures
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.size': 10
})
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd
import earthpy.plot as ep
from getFileName import getFileName, getSaveFileName

#%% Version

version = '1'

#%% Function to visualize classified raster

def visualizeRaster(inpRaster, saveImage=False, saveFileName=None):
    """
    Displays a classified raster with the bands for water, built-up, low-vegetation/soil, and vegetation. It also recognizes cloudmask if it is present in the image. In case the raster corresponds to land-cover change, the increase in water area is highlighted.
    
    Parameters
    ----------
    inpRaster : Numpy or Rasterio array
        The input raster image to display.
    saveImage : Boolean
        Flag to save image to file. Default is False.        

    Returns
    -------
    None.
    """
    
    raster_epsg3857 = inpRaster.rio.reproject('EPSG:3857')
    rasterData = raster_epsg3857.to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    # Determine colormap and titles based on the unique values in the raster
    if np.count_nonzero(np.unique(rasterData)) == 5:
        cMap = geo_cloud_cmap
        titleList = ['Water', 'Built-up', 'Low-vegetation/Soil', 'Vegetation', 'CloudMap']
    elif np.count_nonzero(np.unique(rasterData)) > 5:
        cMap = geo_compare_cmap
        titleDict = {1: 'Water$\\to$Water', 2: 'Water$\\to$Built-up', 3: 'Water$\\to$Low-vegetation', 4: 'Water$\\to$Vegetation', 5: 'Built-up$\\to$Water', 6: 'Built-up$\\to$Built-up', 7: 'Built-up$\\to$Low-vegetation', 8: 'Built-up$\\to$Vegetation', 9: 'Low-vegetation$\\to$Water', 10: 'Low-vegetation$\\to$Built-up', 11: 'Low-vegetation$\\to$Low-vegetation', 12: 'Low-vegetation$\\to$Vegetation', 13: 'Vegetation$\\to$Water', 14: 'Vegetation$\\to$Built-up', 15: 'Vegetation$\\to$Low-vegetation', 16: 'Vegetation$\\to$Vegetation', 251: 'CloudMap'}
        titleList = [titleDict[key] for key in np.unique(rasterData)]
    else:
        cMap = geo_cmap
        titleList = ['Water', 'Built-up', 'Low-vegetation/Soil', 'Vegetation']
    
    im = ax.imshow(rasterData, cmap=cMap, aspect='equal')
    
    # Scale bar
    scalebar = ScaleBar(raster_epsg3857.rio.resolution()[0], 'm', dimension='si-length', color='k', fixed_value=5, fixed_units='km', box_alpha=1, location='lower right', frameon=True, scale_loc='top', label_loc='bottom', sep=3)
    ax.add_artist(scalebar)
    
    # Legend
    lgd = ep.draw_legend(im, titles=titleList, cmap=cMap)
    
    # North arrow
    x, y, arrow_length = -0.05, 0.75, 0.2
    NArrow = ax.annotate('N', xy=(x, y), xytext=(x, y + arrow_length), arrowprops=dict(facecolor='black', arrowstyle='<-', linewidth=3), ha='center', va='center', fontsize=20, xycoords=ax.transAxes)  
    
    plt.show()
    
    # Optionally, save image
    if saveImage:
        if not(saveFileName):
            # Replace 'YOUR_PATH\\TO\\SAVE\\' with the desired directory path
            saveFileName = getSaveFileName('Enter name for the file to save:', initialDir='YOUR_PATH\\TO\\SAVE\\', fileTypes=[('TIFF files', '*.tiff')])
        fig.savefig(''.join([saveFileName.rstrip('tiff'), 'tiff']), format='tiff', bbox_extra_artists=(lgd, NArrow), bbox_inches='tight', dpi=300)
        print(''.join(['Figure has been saved to ', saveFileName]))

#%% Create a custom colormap for visualizing classification later

geo_cmap = LinearSegmentedColormap.from_list('GEO', ['blue', 'darkred', 'goldenrod', 'forestgreen'])
geo_cloud_cmap = LinearSegmentedColormap.from_list('GEO', [(np.interp(1, [1, 251], [0, 1]), 'blue'), (np.interp(2, [1, 251], [0, 1]), 'darkred'), (np.interp(3, [1, 251], [0, 1]), 'goldenrod'), (np.interp(4, [1, 251], [0, 1]), 'forestgreen'), (np.interp(251, [1, 251], [0, 1]), 'darkgrey')])
geo_compare_cmap = LinearSegmentedColormap.from_list('GEO', 
                                                     [(np.interp(1, [1, 251], [0, 1]), 'blue'), 
                                                      (np.interp(1.5, [1, 251], [0, 1]), 'black'), 
                                                      (np.interp(4.5, [1, 251], [0, 1]), 'black'), 
                                                      (np.interp(4.6, [1, 251], [0, 1]), 'aqua'),
                                                      (np.interp(5.5, [1, 251], [0, 1]), 'aqua'), 
                                                      (np.interp(5.6, [1, 251], [0, 1]), 'darkred'), 
                                                      (np.interp(6.5, [1, 251], [0, 1]), 'darkred'),
                                                      (np.interp(6.6, [1, 251], [0, 1]), 'black'),
                                                      (np.interp(8.5, [1, 251], [0, 1]), 'black'),
                                                      (np.interp(8.6, [1, 251], [0, 1]), 'aqua'),
                                                      (np.interp(9.5, [1, 251], [0, 1]), 'aqua'),
                                                      (np.interp(9.6, [1, 251], [0, 1]), 'black'),
                                                      (np.interp(10.5, [1, 251], [0, 1]), 'black'),
                                                      (np.interp(10.6, [1, 251], [0, 1]), 'goldenrod'),
                                                      (np.interp(11.5, [1, 251], [0, 1]), 'goldenrod'), 
                                                      (np.interp(11.6, [1, 251], [0, 1]), 'black'),
                                                      (np.interp(12.5, [1, 251], [0, 1]), 'black'),
                                                      (np.interp(12.6, [1, 251], [0, 1]), 'aqua'),
                                                      (np.interp(13.5, [1, 251], [0, 1]), 'aqua'),
                                                      (np.interp(13.6, [1, 251], [0, 1]), 'black'),
                                                      (np.interp(15.5, [1, 251], [0, 1]), 'black'),
                                                      (np.interp(15.6, [1, 251], [0, 1]), 'forestgreen'), 
                                                      (np.interp(16.1, [1, 251], [0, 1]), 'forestgreen'),
                                                      (np.interp(17, [1, 251], [0, 1]), 'darkgrey'),
                                                      (np.interp(251, [1, 251], [0, 1]), 'darkgrey')])

#%% Get the BFF and AFF classified images

BFF_filename = getFileName('Select the classified before-flash-flood image:', initialDir='YOUR_PATH\\TO\\IMAGES\\Output\\', fileTypes=[('TIF files', '*.tif')])
AFF_filename = getFileName('Select the classified after-flash-flood image:', initialDir='YOUR_PATH\\TO\\IMAGES\\Output\\', fileTypes=[('TIF files', '*.tif')])

with rxr.open_rasterio(AFF_filename) as AFFImg:
    AFFImgData = AFFImg[0]

with rxr.open_rasterio(BFF_filename) as BFFImg:
    BFFImgData = BFFImg[0]

#%% Post processing - Sieve the classified rasters

AFFImgData_sv = rioFeatures.sieve(AFFImgData.to_numpy(), 2)
BFFImgData_sv = rioFeatures.sieve(BFFImgData.to_numpy(), 2)

if 'binary' in os.path.split(AFF_filename)[-1]:
    nonWaterPxIDs = (AFFImgData_sv == 0) & (AFFImgData_sv != 251)
    AFFImgData_sv[nonWaterPxIDs] = BFFImgData_sv[nonWaterPxIDs]

#%% Get comparison image with values from change matrix

comparisonImg = np.zeros(BFFImgData.shape, dtype=np.uint8)
comparisonAreaMat = np.zeros((4, 4))

# Mask the pixels with clouds in either BFF or AFF image with the value 251
combinedCloudMask = (AFFImgData_sv == 251) | (BFFImgData_sv == 251)
comparisonImg[combinedCloudMask] = 251

c1 = 0 # Counter
for BFFClass in range(1, 5):
    for AFFClass in range(1, 5):
        c1 += 1
        tempIDs = (BFFImgData_sv == BFFClass) & (AFFImgData_sv == AFFClass)
        comparisonAreaMat[(BFFClass - 1), (AFFClass - 1)] = np.count_nonzero(tempIDs) # Numpy index goes from 0 to 3 instead of 1 to 4, so subtract 1
        comparisonImg[tempIDs] = c1

# Apply sieve to comparison image
# comparisonImg_sv = rioFeatures.sieve(comparisonImg, 2)
        

#%% Saving metrics

# Convert the pixel data at 20 m x 20 m resolution to sqkm and save as Pandas dataframe
comparisonAreaMat_sqkm_pd = pd.DataFrame(comparisonAreaMat * 400e-6, index=['BFF:Water', 'BFF:Built-up', 'BFF:Low-vegetation', 'BFF:Vegetation'], columns=['AFF:Water', 'AFF:Built-up', 'AFF:Low-vegetation', 'AFF:Vegetation'])
totalClassArea = comparisonAreaMat_sqkm_pd.sum().sum()

AFF_folder, _ = os.path.split(AFF_filename)
BFF_folder, _ = os.path.split(BFF_filename)

# Write land cover change array as raster
comparisonRaster = AFFImgData.copy(data=comparisonImg)
comparisonRaster.rio.to_raster(os.path.join(AFF_folder, 'landCoverChange_v{}.tif'.format(version)))
print(''.join(['Land cover change raster has been saved in the folder ', AFF_folder]))

# Write classified images and land cover change images as TIFF files
visualizeRaster(BFFImgData.copy(data=BFFImgData_sv), saveImage=True, saveFileName=os.path.join(BFF_folder, 'classified_BFF_v{}.tiff'.format(version)))
visualizeRaster(AFFImgData.copy(data=AFFImgData_sv), saveImage=True, saveFileName=os.path.join(BFF_folder, 'classified_AFF_v{}.tiff'.format(version)))
visualizeRaster(comparisonRaster, saveImage=True, saveFileName=os.path.join(AFF_folder, 'landCoverChange_v{}.tiff'.format(version)))

# Write land cover change metrics to a file in AFF output folder
comparisonAreaMat_sqkm_pd.to_csv(os.path.join(AFF_folder, 'landCoverChangeMetrics_sqkm_v{}.txt'.format(version)), mode='w')
with open(os.path.join(AFF_folder, 'landCoverChangeOverview_v{}.txt'.format(version)), 'w') as fID:
    fID.write('--------------------------')
    fID.write('\nBFF file: {}'.format(BFF_filename))
    fID.write('\nAFF file: {}'.format(AFF_filename))
    if 'binary' in os.path.split(AFF_filename)[-1]:
        fID.write('\nNOTE: AFF file contains only water/non-water data. Ignore all other class comparison for BFF --> AFF changes.')
    fID.write('\n--------------------------')
    fID.write('\nTotal classified area (in sq. km) in the images, excluding cloud mask:')
    fID.write('\n{}'.format(totalClassArea))
    fID.write('\n--------------------------\nWater covered area (in sq. km) before flood, excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd.loc['BFF:Water'].sum()))
    fID.write('\nWater covered area fraction before flood, excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd.loc['BFF:Water'].sum() / totalClassArea))
    fID.write('\n--------------------------\nWater covered area (in sq. km) after flood, excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd['AFF:Water'].sum()))
    fID.write('\nWater covered area fraction after flood, excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd['AFF:Water'].sum() / totalClassArea))
    fID.write('\n--------------------------\nIncrease in water covered area (in sq. km) after flood, excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd['AFF:Water'].sum() - comparisonAreaMat_sqkm_pd.loc['BFF:Water'].sum()))
    fID.write('\nIncrease in water covered area fraction after flood, excluding cloud mask:')
    fID.write('\n{}'.format((comparisonAreaMat_sqkm_pd['AFF:Water'].sum() - comparisonAreaMat_sqkm_pd.loc['BFF:Water'].sum()) / totalClassArea))
    fID.write('\nRatio of after-flood water area to before-flood water area:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd['AFF:Water'].sum() / comparisonAreaMat_sqkm_pd.loc['BFF:Water'].sum()))
    fID.write('\n--------------------------\nIncrease in water covered area due to flooding in other land classes (in sq. km), excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd.iloc[1:, 0].sum()))
    fID.write('\nIncrease in water covered area fraction due to flooding in other land classes, excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd.iloc[1:, 0].sum() / totalClassArea))
    fID.write('\n--------------------------\nIncrease in water covered area due to flooding in built-up area (in sq. km), excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd.iloc[1, 0]))
    fID.write('\nContribution of flooding in built-up areas to total excess flooding, excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd.iloc[1, 0] / comparisonAreaMat_sqkm_pd.iloc[1:,0].sum()))
    fID.write('\n--------------------------\nIncrease in water covered area due to flooding in low-vegetation/soil area (in sq. km), excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd.iloc[2, 0]))
    fID.write('\nContribution of flooding in low-vegetation/soil areas to total excess flooding, excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd.iloc[2, 0] / comparisonAreaMat_sqkm_pd.iloc[1:, 0].sum()))
    fID.write('\n--------------------------\nIncrease in water covered area due to flooding in vegetation area (in sq. km), excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd.iloc[3, 0]))
    fID.write('\nContribution of flooding in vegetation areas to total excess flooding, excluding cloud mask:')
    fID.write('\n{}'.format(comparisonAreaMat_sqkm_pd.iloc[3, 0] / comparisonAreaMat_sqkm_pd.iloc[1:, 0].sum()))
print(''.join(['Land cover change metrics have been saved in the folder ', AFF_folder]))