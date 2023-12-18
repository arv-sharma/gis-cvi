# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:34:33 2022
"""
import os
# import rasterio as rio
import rioxarray as rxr
import xarray as xr
# from rasterio.plot import show as rioPlotShow
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.size': 10
#    "font.sans-serif": "Helvetica",
})
import geopandas as gpd
import pandas as pd
from getFileName import getFileName
from sklearn.preprocessing import StandardScaler

#%% Get input image and training file names

# Replace 'your_directory_here' with the actual directory path
trainFileName = getFileName('Select training shape-file:', initialDir='your_directory_here', fileTypes=[('Vector files', '*.shp')])
trainFileFolder, _ = os.path.split(trainFileName)

# Check if training file is for BFF or AFF
inpFileNames = []
if 'BFF' in os.path.split(trainFileName)[-1]:
    inpFileNames.append(getFileName('Select BFF S-1 texture image to be classified:', initialDir='your_directory_here', fileTypes=[('TIF files', '*.tif')]))
    inpFileNames.append(getFileName('Select BFF S-2 image to be classified:', initialDir='your_directory_here', fileTypes=[('TIF files', '*.tif')]))
else:
    inpFileNames.append(getFileName('Select AFF S-1 texture image to be classified:', initialDir='your_directory_here', fileTypes=[('TIF files', '*.tif')]))

#%% Open image and training shapefile, then rearrange data

vector = gpd.read_file(trainFileName)
# Clean up shapefile data to remove nan values from 'LCC' and None values from geometry, and return 4 different LCC's
vector = vector[(vector['LCC'].isin([1, 2, 3, 4]) == True) & ~(vector['geometry'] == None)].reset_index()
if len(inpFileNames) == 1:
    # Return water/not-water LCCs
    vector.loc[vector['LCC'] > 1, 'LCC'] = 2
data_df = pd.DataFrame()
data_df['LCC'] = vector['LCC']
ids2Keep = pd.Series(data=np.full((len(vector),), fill_value=True))
for inpFileName in inpFileNames:
    with rxr.open_rasterio(inpFileName) as dataset:
        bandData = dataset.sel({'x': xr.DataArray(vector.geometry.x, dims='z'), 'y': xr.DataArray(vector.geometry.y, dims='z')}, method='nearest')
    data_df = pd.concat([data_df, pd.DataFrame(bandData.T, columns=bandData.long_name)], axis=1)
    xWithinBounds = abs((vector.geometry.x - bandData.x.values) < (abs(dataset.rio.resolution()[0]) / 2))
    yWithinBounds = abs((vector.geometry.y - bandData.y.values) < (abs(dataset.rio.resolution()[1]) / 2))
    ids2Keep = ids2Keep & xWithinBounds & yWithinBounds

data_df = data_df[ids2Keep].reset_index(drop=True)

#%% Plot spectral signatures for the different classes

colors = ['blue', 'darkred', 'goldenrod', 'forestgreen']
multiClasslabels = ['Water', 'Built-up', 'Low-vegetation/Soil', 'Vegetation']
binaryLabels = ['Water', 'Non-water']
legLabels = binaryLabels if (len(inpFileNames) == 1) else multiClasslabels
numSamples = 25

scaledData = StandardScaler().fit_transform(data_df.filter(regex=('V.*'), axis='columns'))
scaled_df = pd.DataFrame(scaledData, columns=data_df.filter(regex=('V.*'), axis='columns'))
scaled_df['LCC'] = data_df['LCC']
fig, ax = plt.subplots(figsize=(5, 4))
for LCCID in scaled_df['LCC'].unique():
    tempSample = scaled_df.loc[scaled_df['LCC'] == LCCID].sample(numSamples, ignore_index=True).filter(regex=('V.*'), axis='columns')
    ax.plot(tempSample.transpose(), color=colors[int(LCCID) - 1], label=legLabels[int(LCCID) - 1])
ax.legend()
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend([handles[ii] for ii in range(0, data_df['LCC'].unique().size * numSamples, numSamples)], [labels[ii] for ii in range(0, data_df['LCC'].unique().size * numSamples, numSamples)])
plt.xlabel('Metric')
plt.ylabel('Normalized Intensity')
plt.xticks(rotation=30)

# Save figure
# Replace 'your_directory_here' with the actual directory path
saveFileName = os.path.join('your_directory_here', 'SpectralSignatures_S1.tiff')
# fig.savefig(''.join([saveFileName.rstrip('tiff'), 'tiff']), format='tiff', bbox_extra_artists=(lgd, ), bbox_inches='tight', dpi=300)
fig.savefig(''.join([saveFileName.rstrip('tiff'), 'pdf']), format='pdf', bbox_extra_artists=(lgd, ), bbox_inches='tight', dpi=300)
#Note that the bbox_extra_artists must be an iterable
print(''.join(['Figure has been saved to ', saveFileName]))

if len(inpFileNames) == 2:
    fig, ax = plt.subplots(figsize=(5, 4))
    for LCCID in np.sort(data_df['LCC'].unique()):
        tempSample = data_df.loc[data_df['LCC'] == LCCID].sample(numSamples, ignore_index=True).filter(regex=('B.*'), axis='columns')
        ax.plot(tempSample.transpose(), color=colors[int(LCCID) - 1], label=legLabels[int(LCCID) - 1])
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend([handles[ii] for ii in range(0, data_df['LCC'].unique().size * numSamples, numSamples)], [labels[ii] for ii in range(0, data_df['LCC'].unique().size * numSamples, numSamples)])
    plt.xlabel('Band number')
    plt.ylabel('Intensity')
    
    # Save figure
    # Replace 'your_directory_here' with the actual directory path
    saveFileName = os.path.join('your_directory_here', 'SpectralSignatures_S2.tiff')
    # fig.savefig(''.join([saveFileName.rstrip('tiff'), 'tiff']), format='tiff', bbox_extra_artists=(lgd, ), bbox_inches='tight', dpi=300)
    fig.savefig(''.join([saveFileName.rstrip('tiff'), 'pdf']), format='pdf', bbox_extra_artists=(lgd, ), bbox_inches='tight', dpi=300)
    #Note that the bbox_extra_artists must be an iterable
    print(''.join(['Figure has been saved to ', saveFileName]))
