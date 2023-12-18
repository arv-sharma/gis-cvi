# -*- coding: utf-8 -*-
"""
This script calculates the Community Vulnerability Index from geospatial datasets using Factor Analysis.

"""

import rioxarray as rxr
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as skdecomp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
import earthpy.plot as ep

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to plot raster
def plotRaster(raster, categorize=False, saveTiffFileName=None, saveRasterFileName=False, output2Memory=False):
    """
    Plot a raster in EPSG:3857 (WGS 84/Pseudo-Mercator projection). If 'categorize' is True, then the function maps the values in the raster to specific categories.
    Optionally save the projected map as TIFF and save the reprojected raster.

    Parameters
    ----------
    raster : rioxarray.Dataset
        Input raster.
    categorize : bool, optional
        If True, then the raster values will be categorized. The default is False.
    saveTiffFileName : str, optional
        String specifying the name to save the TIFF file. The default is None.
    saveRasterFileName : str, optional
        String specifying the name to save the raster. The default is False.
    output2Memory : str, optional
        Boolean value specifying if the plotted raster data will be returned by the function. The default is False.

    Returns
    -------
    None.

    """
    raster_epsg3857 = raster.rio.reproject('EPSG:3857')
    if categorize:
        raster_data_flat = raster_epsg3857.to_numpy().flatten()
        valid_raster_data = raster_data_flat[~np.isnan(raster_data_flat)]
        bins = valid_raster_data.std() * np.asarray([-np.inf, -1, -0.5, 0.5, 1])  # 5 maps to CVI_raw >= CVI_raw.std()
        cat_raster_data_flat = np.digitize((valid_raster_data - valid_raster_data.mean()), bins=bins)
        cat_raster_data_mat = np.full(raster_epsg3857.shape, fill_value=251, dtype='uint8')
        cat_raster_data_mat[~np.isnan(raster_epsg3857)] = cat_raster_data_flat
        raster2Plot = raster_epsg3857.copy(data=cat_raster_data_mat)
        titleDict = {1: 'Very low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very high', 251: 'No data'}
        titleList = [titleDict[key] for key in np.unique(cat_raster_data_mat)]
        # Colormap
        pxMinMax = [cat_raster_data_mat.min(), cat_raster_data_mat.max()]
        cmapDict = {1: 'palegoldenrod', 2: 'gold', 3: 'orange', 4: 'orangered', 5: 'darkred', 251: 'gainsboro'}
        geo_cat_cmap = LinearSegmentedColormap.from_list('GEO', [(np.interp(number, pxMinMax, [0, 1]), cmapDict[number]) for number in np.unique(cat_raster_data_mat)])
        cMap = geo_cat_cmap
    else:
        raster2Plot = raster_epsg3857
        cMap = 'inferno'
        
    # Plot
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    im = ax.imshow(raster2Plot.to_numpy(), cmap=cMap, aspect='equal')
    # Scale bar
    scalebar = ScaleBar(raster_epsg3857.rio.resolution()[0], 'm', dimension='si-length', color='k', fixed_value=5, fixed_units='km', box_alpha=1, location='lower right', frameon=True, scale_loc='top', label_loc='bottom', sep=3)
    ax.add_artist(scalebar)
    # North arrow
    x, y, arrow_length = -0.05, 0.75, 0.2
    NArrow = ax.annotate('N', xy=(x, y), xytext=(x, y + arrow_length), arrowprops=dict(facecolor='black', arrowstyle='<-', linewidth=3), ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    # Legend or colorbar
    if categorize:
        lgd = ep.draw_legend(im, titles=titleList, cmap=cMap)
    else:
        cb = plt.colorbar(im)
    plt.show()
    
    # Optionally, save image
    if saveTiffFileName:
        if categorize:
            fig.savefig(''.join([saveTiffFileName.rstrip('.tiff'), '.tiff']), format='tiff', bbox_extra_artists=(lgd, NArrow), bbox_inches='tight', dpi=300)
        else:
            fig.savefig(''.join([saveTiffFileName.rstrip('.tiff'), '.tiff']), format='tiff', bbox_extra_artists=(NArrow,), bbox_inches='tight', dpi=300)
            
    # Optionally, save raster
    if saveRasterFileName:
        if categorize:
            raster2Plot.rio.set_nodata(251)
        raster2Plot.rio.to_raster(saveRasterFileName)
    
    plt.close()
    
    # Optionally, return plotted raster data
    if output2Memory:
        return(raster2Plot)

# Function to apply Factor Analysis (FA)
def applyFA(df, fullTgtFolderPath, n_factors=None, rotation='varimax', method='ml', use_smc=True):    
    # Calculate VIF to find multicollinearity
    vif_df = pd.DataFrame()
    vif_df.index.name = 'Feature ID'
    vif_df['feature'] = df.columns
    exitFlag = False
    count = 0
    while not(exitFlag):
        # Scale the data with zero-mean and unit variance
        scaledData = StandardScaler().fit_transform(df)  
        # Factor analysis metrics
        chi_square_value, p_value = calculate_bartlett_sphericity(scaledData)
        print('Chi square value: {}\np-value: {}\n'.format(chi_square_value, p_value))
        kmo_all, kmo_model = calculate_kmo(scaledData)
        print('KMO metric: {}'.format(kmo_model))
        # VIF
        for idx, column in enumerate(df.columns):
            vif_df.loc[vif_df.index[vif_df['feature'] == column], 'vif_v{}'.format(count)] = variance_inflation_factor(scaledData, idx)
        print('The variance inflation factors are:\n\n{}'.format(vif_df))
        facIdx2Drop_str = input('Enter the feature IDS (0 to {}) to drop, separated by comma(type n if nothing is required to drop): '.format(len(vif_df) - 1))
        if facIdx2Drop_str == 'n':
            exitFlag = True
            break
        # Check if the inputs are valid integers
        facIdx2Drop = np.array([int(item) for item in facIdx2Drop_str.split(',')])
        assert (facIdx2Drop <= (len(vif_df) - 1)).all() and (facIdx2Drop >= 0).all(), 'The indices to drop are not valid.'
        df.drop(columns=vif_df.loc[facIdx2Drop, 'feature'], inplace=True)
        count += 1
    
    
    # Factor analysis
    if not n_factors:
        n_factors = len(df.columns)
    else:
        n_factors = int(n_factors)  # To get integer value from *args string
    if use_smc or (use_smc == 'True'):
        use_smc = True
    else:
        use_smc = False
    
    FA = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method, use_smc=use_smc)
    FA.fit(scaledData)
    
    # Scree plot
    fig, ax = plt.subplots(figsize=(4.5, 4))
    plt.plot(np.arange(1, len(df.columns) + 1), FA.get_eigenvalues()[0], 'ko-')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('Factor')
    plt.ylabel('Eigenvalue')
    fig.savefig(''.join([os.path.join(fullTgtFolderPath, 'ScreePlot'), '.tiff']), format='tiff', dpi=300)
    
    # FA metrics
    FA_loadings_df = pd.DataFrame(FA.loadings_, columns=[''.join(('Factor', str(factorNum))) for factorNum in range(1, n_factors + 1)], index=df.columns)
    FA_weights_df = np.sign(FA_loadings_df) * (FA_loadings_df ** 2) / (FA_loadings_df ** 2).sum(axis=0)
    FA_df = FA_loadings_df.merge(FA_weights_df, on=FA_loadings_df.index, suffixes=['_loading', '_weight'])
    FA_explVarRatio = FA.get_eigenvalues()[0][:n_factors] / FA.get_eigenvalues()[0][:n_factors].sum()
    # Write FA metrics to file
    FA_df.to_csv(os.path.join(fullTgtFolderPath, 'FA_metrics' + '.txt'), mode='w')
    with open(os.path.join(fullTgtFolderPath, 'FA_metrics' + '.txt'), 'a') as fID:
        fID.write('--------------------------')
        fID.write('\nBartlett sphericity test')
        fID.write('\nChi-square value: {}'.format(chi_square_value))
        fID.write('\np-value: {}'.format(p_value))
        fID.write('\n--------------------------')
        fID.write('\nKMO Model: {}'.format(kmo_model))
        fID.write('\n--------------------------')
        fID.write('\nFA n_factors: {}'.format(n_factors))
        fID.write('\nFA Rotation: {}'.format(rotation))
        fID.write('\nFA Method: {}'.format(method))
        fID.write('\nFA use_smc: {}'.format(use_smc))
        fID.write('\n--------------------------')
        fID.write('\nEigenvalues: {}'.format(FA.get_eigenvalues()[0]))
        fID.write('\nPercentage of variance of raw-data explained by factor: {}'.format(FA.get_eigenvalues()[0] / FA.get_eigenvalues()[0].sum()))
        fID.write('\nExtracted components explain {} of the variance in the raw-data.'.format((FA.get_eigenvalues()[0] / FA.get_eigenvalues()[0].sum())[:n_factors].sum()))
        fID.write('\nExplained variance ratio of extracted factors: {}'.format(FA_explVarRatio))
        fID.write('\n--------------------------')
        fID.write('\nVariance Inflation Factor (VIF):\n{}'.format(vif_df))
        
    
    # CVI
    intermediateCI = np.matmul(scaledData, FA_weights_df.to_numpy()) # Intermediate composite index
    CVI_raw = np.matmul(intermediateCI, FA_explVarRatio)
    print('Eigenvalues: {}'.format(FA.get_eigenvalues()[0]))
    return(CVI_raw, FA_loadings_df, FA_weights_df)

# Function to calculate Composite Vulnerability Index (CVI) for each city
def calcCVI(rootFolder, targetFolder, folder, cleanVariables=False):
    tifFiles = glob.glob(os.path.join(rootFolder, folder, 'derived', '*.tif'))
    # Remove GDP increase raster since its data is limited
    tifFiles = [file for file in tifFiles if not ('GDPIncrease.tif' in file)]
    # Check if target folder path exists
    fullTgtFolderPath = os.path.join(targetFolder, folder)
    if not os.path.exists(fullTgtFolderPath):
        os.makedirs(fullTgtFolderPath)
    # Get the non-nan pixel union from all datasets
    for fileNum, fileName in enumerate(tifFiles):
        with rxr.open_rasterio(fileName) as srcRaster:
            if fileNum == 0:
                validPx = ~np.isnan(srcRaster[0])
            else:
                validPx = validPx & ~np.isnan(srcRaster[0])
    # Build a dataframe with valid pixels and save raster
    rawData_df = pd.DataFrame()
    for fileName in tifFiles:
        fileID = os.path.split(fileName)[-1].split('.')[0]
        with rxr.open_rasterio(fileName) as srcRaster:
            srcRasterData = srcRaster[0]
        plotRaster(srcRasterData, categorize=False, saveTiffFileName=os.path.join(fullTgtFolderPath, fileID + '_epsg3857'), saveRasterFileName=os.path.join(fullTgtFolderPath, fileID + '_epsg3857' + '.tif'))
        plotRaster(srcRasterData, categorize=True, saveTiffFileName=os.path.join(fullTgtFolderPath, fileID + '_cat_epsg3857'), saveRasterFileName=os.path.join(fullTgtFolderPath, fileID + '_cat_epsg3857' + '.tif'))
        rawData_df[fileID] = srcRasterData.to_numpy().flatten()[validPx.to_numpy().flatten()]
        
    # Combine a0to14 and a65plus columns
    rawData_df['popDens_a0to14_65plus_norm'] = rawData_df['popDens_a0to14_norm'] + rawData_df['popDens_a65plus_norm']
    rawData_df = rawData_df.drop(columns=['popDens_a0to14_norm', 'popDens_a65plus_norm'])
    # Optionally, drop popDens columns for SilcharIND
    # if folder == 'SilcharIND':
    #     rawData_df = rawData_df.drop(columns=['popDensIncrease', 'popDens_a0to14_65plus_norm'])
        
    # Write raw data to file
    rawData_df.to_csv(os.path.join(fullTgtFolderPath, 'RawData.txt'), mode='w')
        
    # Apply 0 to 1 scale for all data
    normData_df = pd.DataFrame()
    for column in rawData_df.columns:
        normData_df[column] = (rawData_df[column] - rawData_df[column].min()) / (rawData_df[column].max() - rawData_df[column].min())
        if column in ['GDPIncrease', 'infra_OSM']:
            normData_df[column] = 1 - normData_df[column]
            
    # CVI
    plt.close('all')
    isSatisfactory = False
    while not isSatisfactory:
        FA_args = list(input('Enter optional arguments for FA in the following order, seperated by spaces: n_factors rotation method use_smc . If you want to proceed with default option, enter "default" as first option: ').strip().split())
        if FA_args[0] == 'default':
            CVI_raw, FA_loadings_df, FA_weights_df = applyFA(normData_df.copy(), fullTgtFolderPath, n_factors=None, rotation='varimax', method='ml', use_smc=True)
        else:
            CVI_raw, FA_loadings_df, FA_weights_df = applyFA(normData_df.copy(), fullTgtFolderPath, *FA_args)
        print('FA loadings: \n{}\n'.format(FA_loadings_df))
        print('FA weights: \n{}\n\n\n'.format(FA_weights_df))
        isSatisfactory = int(input('Enter 1 if the results look good, else 0: '))        
        
    CVI_raw_mat = np.full(srcRasterData.shape, np.nan)
    CVI_raw_mat[validPx] = CVI_raw
      
    # CVI as raster
    CVI_raw_raster = srcRasterData.copy(data=CVI_raw_mat)
    # CVI_raw_raster.rio.set_nodata(np.nan)
    CVI_raw_raster.rio.to_raster(os.path.join(fullTgtFolderPath, 'CVI_raw_epsg4326.tif'))

    plotRaster(CVI_raw_raster, categorize=False, saveTiffFileName=os.path.join(fullTgtFolderPath, 'CVI_raw_epsg3857'), saveRasterFileName=os.path.join(fullTgtFolderPath, 'CVI_raw_epsg3857' + '.tif'))
    plotRaster(CVI_raw_raster, categorize=True, saveTiffFileName=os.path.join(fullTgtFolderPath, 'CVI_cat_epsg3857'), saveRasterFileName=os.path.join(fullTgtFolderPath, 'CVI_cat_epsg3857' + '.tif'))
    
    print('\nCompleted processing data for ' + folder)
    plt.close('all')

# Main function
if __name__ == '__main__':
    rootFolder = input('Enter the root folder address:')
    targetFolder = input('Enter the target folder address: ')
    for folder in os.listdir(rootFolder):
        calcCVI(rootFolder, targetFolder, folder)
    # Parallel(n_jobs=cpu_count())(delayed(calcCVI)(rootFolder, targetFolder, folder, cleanVariables=cleanVariables) for folder in tqdm(os.listdir(rootFolder), total=len(os.listdir(rootFolder)), desc='Progress'))
