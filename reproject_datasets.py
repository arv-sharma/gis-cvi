# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:12:48 2022

This script reprojects and saves several datasets based on a Sentinel-2 image's bounds.

"""

import rioxarray as rxr
import xarray as xr
import numpy as np
import glob
import os

#%% Simple raster reproject

def raster_reproject(src, inpFile_raw, outFile):
    """
    Function to reproject a raw input file to the bounds and CRS specified in target.
    The output is saved along with a copy where the values corresponding to NaN pixels
    and water pixels in target are set to NaN.

    Parameters
    ----------
    src : str
        Source dataset/dataarray whose CRS and bounds will be used for reprojection.
    inpFile_raw : str
        Filepath of GeoTIFF file whose bounds and CRS will be modified during reprojection.
    outFile : str
        Filepath for writing output file.

    Returns
    -------
    None.

    """
    with rxr.open_rasterio(inpFile_raw) as dst_raw:
        dst_clipped = dst_raw.rio.clip_box(*src.rio.bounds())
        
    # noDataVal = np.nan if dst_clipped.dtype == 'float' else 251
    dst_repr = dst_clipped.rio.reproject_match(src)
            
    if dst_repr.shape[0] != 1:
        raise ValueError('More than one band in the dataset.')
    
    # dst_repr.rio.set_nodata(noDataVal)
    dst_repr.rio.to_raster(outFile)
    # fig, ax = plt.subplots()
    # dst_repr_masked[0].plot.imshow(ax=ax)
    # dst_repr_masked[0].plot.imshow(ax=ax)
    # x_clipped = dst_raw.x[((dst_raw.x + dst_raw.rio.resolution()[0] / 2) >= src.rio.bounds()[0]) & ((dst_raw.x - dst_raw.rio.resolution()[0] / 2) <= src.rio.bounds()[2])]
    # y_clipped = dst_raw.y[((dst_raw.y - dst_raw.rio.resolution()[1] / 2) >= src.rio.bounds()[1]) & ((dst_raw.y + dst_raw.rio.resolution()[1] / 2) <= src.rio.bounds()[3])]
    
#%% Main function

def main():
    
    # Replace 'path\to\your\input\data' with the actual path to your input data
    srcFilePaths = glob.glob(os.path.join(r'path\to\your\input\data', '*S2_BFF*'))
    
    # Replace the addresses with the actual paths to your datasets
    datasets = {
        'CI': {'address': r'path\to\your\CI.tif'},
        'SMOD': {'address': r'path\to\your\SMOD.tif'},
        'lulc_human_mod': {'address': r'path\to\your\lulc_human_mod.tif'},
        'popDens_a0to14': {'address': r'path\to\your\popDens_a0to14.tif'},
        'popDens_a65plus': {'address': r'path\to\your\popDens_a65plus.tif'},
        'popDens_female': {'address': r'path\to\your\popDens_female.tif'},
        'popDens_all': {'address': r'path\to\your\popDens_all.tif'},
        'popDens_2000': {'address': r'path\to\your\popDens_2000.tif'},
        'popDens_2005': {'address': r'path\to\your\popDens_2005.tif'},
        'popDens_2010': {'address': r'path\to\your\popDens_2010.tif'},
        'popDens_2015': {'address': r'path\to\your\popDens_2015.tif'},
        'popDens_2020': {'address': r'path\to\your\popDens_2020.tif'},
        'GDP': {'address': r'path\to\your\GDP.nc'},
        'SHDI': {'address': r'path\to\your\SHDI.tif'},
        'GRDI': {'address': r'path\to\your\GRDI.tif'}
    }
    
    for item in srcFilePaths:
        if not('BethaniaAUS' in item):
            continue
        # Get the city and country name to make folders
        ID = os.path.split(item)[-1].lstrip('GEE_').split('_S2')[0]
        targetRoot = os.path.join(r'path\to\your\target\root', ID)
        if not os.path.exists(targetRoot):
            os.makedirs(targetRoot)
        
        srcFile = os.path.join(item, 'classified_' + ID + '_BFF.tif')
        
        with rxr.open_rasterio(srcFile) as src:
            with rxr.open_rasterio(datasets['popDens_all']['address']) as dst_raw:
                dst_clipped_gpw = dst_raw.rio.clip_box(*src.rio.bounds())
        
        dst_clipped_cleaned_gpw = dst_clipped_gpw.where(dst_clipped_gpw >= 0, other=np.nan)
        dst_clipped_cleaned_gpw.rio.to_raster(os.path.join(targetRoot, 'popDens_all' + '.tif'))        
        
        for datasetID, datasetMeta in datasets.items():
            if datasetID == 'popDens_all':
                continue
            
            inpFile_raw = datasetMeta['address']
            
            if datasetID == 'GDP':
                with xr.open_dataset(inpFile_raw, engine='netcdf4') as dst_raw:
                    dst_raw_data = dst_raw.GDP_PPP
                
                for bandNum in range(dst_raw_data.shape[0]):
                    dst_raw_data_slice = dst_raw_data[bandNum]
                    dst_raw_data_slice.rio.write_crs(4326, inplace=True)
                    dst_raw_data_slice_clipped = dst_raw_data_slice.rio.clip_box(*src.rio.bounds())
                    dst_repr = dst_raw_data_slice_clipped.rio.reproject_match(dst_clipped_cleaned_gpw)
                    outFile = os.path.join(targetRoot, datasetID + str(int(dst_raw_data_slice.time.values)) + '.tif')
                    dst_repr.rio.to_raster(outFile)
            else:                
                outFile = os.path.join(targetRoot, datasetID + '.tif')
                raster_reproject(dst_clipped_cleaned_gpw, inpFile_raw, outFile)
        
        print('Completed saving datasets for {}.'.format(ID))

#%% Run main function

if __name__ == '__main__':
    main()
