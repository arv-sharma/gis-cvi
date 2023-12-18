# -*- coding: utf-8 -*-
"""
Script to process and analyze data related to Composite Vulnerability Index (CVI) across multiple sites used in the study.
"""

import rioxarray as rxr
import os
from localCVI_FA import plotRaster, applyFA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
import earthpy.plot as ep

#%% Main

if __name__ == '__main__':
    # Define root and target folders
    rootFolder = r'path/to/your/root/folder'  # Replace with the actual root folder path
    targetFolder = os.path.join(rootFolder, 'AllCities')
    
    # Create target folder if it does not exist
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
        
    # Get a list of folders in the root directory
    folders = [item for item in os.listdir(rootFolder) if os.path.isdir(os.path.join(rootFolder, item))]
    
    data_dict = {}
    data_df = pd.DataFrame()
    
    # Loop through each city folder
    for folder in folders:
        if folder != 'AllCities':
            # Read raw raster data
            with rxr.open_rasterio(os.path.join(rootFolder, folder, 'CVI_raw_epsg4326.tif')) as raster:
                rasterData = raster[0]
            
            # Read city data from CSV
            cityData_df = pd.read_csv(os.path.join(rootFolder, folder, 'RawData.txt'), index_col=0)
            
            # Read categorized raster data
            with rxr.open_rasterio(os.path.join(rootFolder, folder, 'CVI_cat_epsg3857.tif')) as raster:
                data_dict[folder] = [rasterData, ~np.isnan(rasterData.to_numpy()), len(data_df), len(data_df) + len(cityData_df), raster[0]]
            
            # Concatenate city data to the overall dataframe
            data_df = pd.concat([data_df, cityData_df], ignore_index=True)
            
    # Write raw data to file
    data_df = data_df.dropna(axis=1)
    data_df.to_csv(os.path.join(targetFolder, 'RawData.txt'), mode='w')
        
    # Apply 0 to 1 scale for all data
    normData_df = pd.DataFrame()
    for column in data_df.columns:
        normData_df[column] = (data_df[column] - data_df[column].min()) / (data_df[column].max() - data_df[column].min())
        if column in ['GDP2015', 'GDPIncrease', 'infra_OSM']:
            normData_df[column] = 1 - normData_df[column]
            
    # CVI
    isSatisfactory = False
    while not isSatisfactory:
        # User input for optional Factor Analysis (FA) arguments
        FA_args = list(input('Enter optional arguments for FA in the following order, separated by spaces: n_factors rotation method use_smc. If you want to proceed with default options, enter "default" as the first option: ').strip().split())
        
        if FA_args[0] == 'default':
            CVI_raw, FA_loadings_df, FA_weights_df = applyFA(normData_df, targetFolder, n_factors=None, rotation='varimax', method='ml', use_smc=True)
        else:
            CVI_raw, FA_loadings_df, FA_weights_df = applyFA(normData_df, targetFolder, *FA_args)
            
        print('FA loadings: {}'.format(FA_loadings_df))
        print('FA weights: {}\n\n\n'.format(FA_weights_df))
        isSatisfactory = int(input('Enter 1 if the results look good, else 0: '))        
    
    bins = CVI_raw.std() * np.asarray([-np.inf, -1, -0.5, 0.5, 1])
    CVI_cat = np.digitize(CVI_raw, bins=bins)
      
    # CVI as raster and output data
    diff_df = pd.DataFrame(index=data_dict.keys(), columns=['MaxDiff' + str(num) for num in range(0, 4)])
    
    for key, val in data_dict.items():
        # Local cat CVI and global raw CVI
        cityCVI_flat = CVI_raw[val[2]:val[3]]
        cityCVI_mat = np.full(val[1].shape, fill_value=np.nan)
        cityCVI_mat[val[1]] = cityCVI_flat
        cityCVI_raster = val[0].copy(data=cityCVI_mat)
        plotRaster(cityCVI_raster, categorize=False, saveTiffFileName=os.path.join(targetFolder, key + '_globalCVI_raw_epsg3857'), saveRasterFileName=os.path.join(targetFolder, key + '_localCVI_raw_epsg3857.tif'))
        cityCVI_cat_raster = plotRaster(cityCVI_raster, categorize=True, saveTiffFileName=os.path.join(targetFolder, key + '_localCVI_cat_epsg3857'), saveRasterFileName=os.path.join(targetFolder, key + '_localCVI_cat_epsg3857.tif'), output2Memory=True)
        validIDs = (cityCVI_cat_raster.to_numpy() != 251)
        cityCVI_cat_absDiff = abs(cityCVI_cat_raster.to_numpy().flatten().astype('float64')[validIDs.flatten()] - val[-1].to_numpy().flatten().astype('float64')[validIDs.flatten()])
        
        for diff in range(0, 4):
            diff_df.at[key, 'MaxDiff' + str(diff)] = np.count_nonzero(cityCVI_cat_absDiff <= diff) / np.count_nonzero(validIDs)
        
        # Global CVI
        globalCityCVI_flat = CVI_cat[val[2]:val[3]]
        globalCityCVI_mat = np.full(val[1].shape, fill_value=251, dtype='uint8')
        globalCityCVI_mat[val[1]] = globalCityCVI_flat
        globalCityCVI_raster = val[0].copy(data=globalCityCVI_mat)
        globalCityCVI_raster.rio.set_nodata(251)
        raster2Plot = globalCityCVI_raster.rio.reproject('EPSG:3857')
        
        # Plot
        titleDict = {1: 'Very low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very high', 251: 'No data'}
        titleList = [titleDict[keyNum] for keyNum in np.unique(raster2Plot)]
        
        # Colormap
        pxMinMax = [raster2Plot.min(), raster2Plot.max()]
        cmapDict = {1: 'palegoldenrod', 2: 'gold', 3: 'orange', 4: 'orangered', 5: 'darkred', 251: 'gainsboro'}
        geo_cat_cmap = LinearSegmentedColormap.from_list('GEO', [(np.interp(number, pxMinMax, [0, 1]), cmapDict[number]) for number in np.unique(raster2Plot)])
        cMap = geo_cat_cmap
        
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        im = ax.imshow(raster2Plot.to_numpy(), cmap=cMap, aspect='equal')
        
        # Scale bar
        scalebar = ScaleBar(raster2Plot.rio.resolution()[0], 'm', dimension='si-length', color='k', fixed_value=5, fixed_units='km', box_alpha=1, location='lower right', frameon=True, scale_loc='top', label_loc='bottom', sep=3)
        ax.add_artist(scalebar)
        
        # North arrow
        x, y, arrow_length = -0.05, 0.75, 0.2
        NArrow = ax.annotate('N', xy=(x, y), xytext=(x, y + arrow_length), arrowprops=dict(facecolor='black', arrowstyle='<-', linewidth=3), ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
        
        # Legend or colorbar
        lgd = ep.draw_legend(im, titles=titleList, cmap=cMap)
        plt.show()
        
        fig.savefig(os.path.join(targetFolder, key + '_globalCVI_cat_epsg3857.tiff'), format='tiff', bbox_extra_artists=(lgd, NArrow), bbox_inches='tight', dpi=300)
        plt.close()
        raster2Plot.rio.to_raster(os.path.join(targetFolder, key + '_globalCVI_cat_epsg3857.tif'))
        
    # Write difference in global vs. local categorization data to file
    diff_df.to_csv(os.path.join(targetFolder, 'global_vs_local_diff.txt'), mode='w')
