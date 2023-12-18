# -*- coding: utf-8 -*-
"""
Created on Sat May 7 14:33:36 2022
"""

# -*- coding: utf-8 -*-

import os
import rioxarray as rxr
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import earthpy.plot as ep
import xgboost as xgb
from getFileName import getFileName, getSaveFileName

#%% Function to classify raster based on training file

def classify_raster(case_id, is_binary):
    """
    This function receives a case_id that identifies the location and after-flood or before-flood image (AFF or BFF). 
    It then asks the user for the locations of input TIF files to be trained and the training shape-file. 
    It saves training data, such as the number of labels, in the parent folder of the training shape-file. 
    It uses xgboost to train the classifier and saves validation metrics in the training folder. 
    Finally, it asks the user for a file name to save the classified image. It also saves the training data as a raster in the training folder.

    Parameters
    ----------
    case_id : String
        A string to identify the case that is being classified. Example: 'BethaniaAUS_BFF'.
    is_binary : bool
        A flag indicating whether the classification is binary or multi-class.

    Returns
    -------
    None.

    """
    
    #%% Get input images and training file names
    s1_inp_file_name = getFileName('Select S-1 ' + case_id + ' image to classify:', initialDir='your_initial_directory_here', fileTypes=[('TIF files', '*.tif')])
    s2_inp_file_name = getFileName('Select S-2 ' + case_id + ' image to classify:', initialDir='your_initial_directory_here', fileTypes=[('TIF files', '*.tif')])
    train_file_name = getFileName('Select ' + case_id + ' training shape-file to classify:', initialDir='your_initial_directory_here', fileTypes=[('Vector files', '*.shp')])
    
    # If the user hits cancel on any of the input files, exit the function
    if (s1_inp_file_name == '') or (s2_inp_file_name == '') or (train_file_name == ''):
        print('Skipping processing the case: ' + case_id)
        return
    
    #%% Open training shapefile, then rearrange data
    vector = gpd.read_file(train_file_name)
    # Clean up shapefile data to remove nan values from 'LCC' and None values from geometry, and return 4 different LCC's
    vector = vector[(vector['LCC'].isin([1, 2, 3, 4]) == True) & ~(vector['geometry'] == None)].reset_index()
    inp_file_names = [s1_inp_file_name]
    if is_binary:
        # Return water/not-water LCCs
        vector.loc[vector['LCC'] > 1, 'LCC'] = 2
    else:
        inp_file_names.append(s2_inp_file_name)
    data_df = pd.DataFrame()
    data_df['LCC'] = vector['LCC']
    ids2Keep = pd.Series(data=np.full((len(vector),), fill_value=True))
    for inp_file_name in inp_file_names:
        with rxr.open_rasterio(inp_file_name) as dataset:
            band_data = dataset.sel({'x': xr.DataArray(vector.geometry.x, dims='z'), 'y': xr.DataArray(vector.geometry.y, dims='z')}, method='nearest')
            # Remove the VV and VH correlation bands from texture data since they don't improve classification. See spectral signatures/band visualization
        if 'dB_texture' in os.path.split(inp_file_name)[-1]:
            band_data = band_data.drop_sel(band=[5, 8])

        data_df = pd.concat([data_df, pd.DataFrame(band_data.T, columns=[name for name in band_data.long_name if not('correlation' in name)])], axis=1)
        x_within_bounds = abs((vector.geometry.x - band_data.x.values) < (abs(dataset.rio.resolution()[0]) / 2))
        y_within_bounds = abs((vector.geometry.y - band_data.y.values) < (abs(dataset.rio.resolution()[1]) / 2))
        ids2Keep = ids2Keep & x_within_bounds & y_within_bounds

    data_df = data_df[ids2Keep].reset_index(drop=True)
        
    #%% Plot the images to be used in classification
    
    with rxr.open_rasterio(s2_inp_file_name) as s2_imgs:
        with rxr.open_rasterio(s1_inp_file_name) as s1_imgs:
            s1_imgs_filtered = s1_imgs.drop_sel(band=[5, 8])
            s1_imgs_repr = s1_imgs_filtered.rio.reproject_match(s2_imgs, nodata=np.nan)
    
    ## Plot all bands
    if is_binary:
        img_stack = s1_imgs_repr.to_numpy()
    else:
        img_stack = np.concatenate((s1_imgs_repr.to_numpy(), s2_imgs.to_numpy()), axis=0)

    ep.plot_bands(img_stack, cmap='gist_earth', figsize=(18, 12), cols=(3 if is_binary else 5), cbar=False)
    
    #%% Classification preprocessing
    
    # Can skip (***).to_numpy() if desired. It is used here for a consistent representation of data and labels
    X_data = data_df.filter(regex=("V.*|B.*"), axis='columns').copy().to_numpy()
    y_data = data_df['LCC'].copy().to_numpy()
    if is_binary:
        y_data[y_data != 1] = 0

    scaler = StandardScaler().fit(X_data)
    X_data_scaled = scaler.transform(X_data)

    # Train-test split (25% to test)
    X_data_scaled_train, X_data_scaled_test, y_data_train, y_data_test = train_test_split(X_data_scaled, y_data, test_size=0.25, stratify=y_data)

    print(f"X train shape: {X_data_scaled_train.shape}\nX test Shape: {X_data_scaled_test.shape}\ny train Shape: {y_data_train.shape}\ny test Shape:{y_data_test.shape}")
    
    #%% XGBoost classification
    
    if is_binary:
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=np.ceil(np.sum(y_data == 0) / np.sum(y_data == 1))) # scale_pos_weight increases the penalty on misclassifying the minor class (which is water here)
    else:
        xgb_model = xgb.XGBClassifier(objective="multi:softprob")
    # xgb_model = xgb.XGBClassifier(objective="binary:logistic")
    y_data_train_xgb = y_data_train.astype('int')
    y_data_test_xgb = y_data_test.astype('int')

    # Classify data with Extreme Gradient Boosting classifier
    xgb_model.fit(X_data_scaled_train, y_data_train_xgb)

    # Predict for test data
    xgb_pred = xgb_model.predict(X_data_scaled_test)

    # Accuracy and Classification Report
    print(f"Accuracy: {accuracy_score(y_data_test_xgb, xgb_pred) * 100}%")
    # print(classification_report(y_data_test_xgb, xgb_pred, target_names=['Water (1)', 'Others (2)']))
    # print(confusion_matrix(y_data_test_xgb, xgb_pred, labels=[1, 2]))
    print(classification_report(y_data_test_xgb, xgb_pred, target_names=(['Non-water (0)', 'Water (1)'] if is_binary else ['Water (1)', 'Built-up (2)', 'Low-veg/soil (3)', 'Veg (4)'])))
    print(confusion_matrix(y_data_test_xgb, xgb_pred, labels=([0, 1] if is_binary else [1, 2, 3, 4])))
    
    #%% Write training metrics to file
    train_file_folder, _ = os.path.split(train_file_name)
    # Write training file label counts to a file in the training folder
    data_df['LCC'].value_counts().to_csv(os.path.join(train_file_folder, 'trainDataOverview.txt'), mode='w', header=['Label_count'])
    data_df['LCC'].value_counts(normalize=True).to_csv(os.path.join(train_file_folder, 'trainDataOverview.txt'), mode='a', header=['Normalized_Label_count'])
    with open(os.path.join(train_file_folder, 'trainDataOverview.txt'), 'a') as f_id:
        f_id.write('\n\n--------------------------')
        if is_binary:
            f_id.write('\nClassification type: binary')
            f_id.write('\nInput S-1 file: {}'.format(s1_inp_file_name))
        else:
            f_id.write('\nClassification type: multi')
            f_id.write('\nInput S-1 file: {}'.format(s1_inp_file_name))
            f_id.write('\nInput S-2 file: {}'.format(s2_inp_file_name))
        f_id.write('\n\n--------------------------')
        f_id.write('\nConfusion matrix')
        f_id.write('\n' + str(confusion_matrix(y_data_test_xgb, xgb_pred, labels=([0, 1] if is_binary else [1, 2, 3, 4]))))
        f_id.write('\n\n--------------------------')
        f_id.write('\nClassification report\n')
    _ = classification_report(y_data_test_xgb, xgb_pred, output_dict=True, target_names=(['Non-water (0)', 'Water (1)'] if is_binary else ['Water (1)', 'Built-up (2)', 'Low-veg/soil (3)', 'Veg (4)']))
    pd.DataFrame(_).transpose().to_csv(os.path.join(train_file_folder, 'trainDataOverview.txt'), mode='a')
    
    #%% Apply classification to original raster bands and predict
    
    # The image array initially a stack of N rows, where N is the band number. 
    # Change the ordering to the image stack of size N. Then reshape to an array with N columns
    img_X_data = np.moveaxis(img_stack, 0, -1).reshape(-1, X_data.shape[-1])
    # s1s2Img_X_data = s1s2Img_array.reshape(-1, s1Img.count + s2Img.count)
    img_X_data_scaled = scaler.transform(img_X_data)
    # Get Nan IDs
    nan_ids = np.isnan(img_X_data_scaled).any(axis=1)
    if not is_binary:
        s2_img_data = np.moveaxis(s2_imgs.to_numpy(), 0, -1).reshape(-1, s2_imgs.shape[0])
        nan_ids = nan_ids | (s2_img_data == 0).all(axis=1)
    # Assign values
    classified_img_no_nan = xgb_model.predict(img_X_data_scaled[~nan_ids])
    classified_img = np.full((s2_imgs[0].size, ), fill_value=np.nan)
    classified_img[~nan_ids] = classified_img_no_nan
    classified_img = classified_img.reshape(s2_imgs[0].shape)
    classified_img_for_raster = np.full((s2_imgs[0].size, ), fill_value=251, dtype='uint8')
    classified_img_for_raster[~nan_ids] = classified_img_no_nan
    classified_img_for_raster = classified_img_for_raster.reshape(s2_imgs[0].shape)
    
    #%% Visualize classified map

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    im = ax.imshow(classified_img, cmap=(geo_cmap_binary if is_binary else geo_cmap_multi))
    ep.draw_legend(im, titles=(['Not-water', 'Water'] if is_binary else ['Water', 'Built-up', 'Low-vegetation/Soil', 'Vegetation']), cmap=(geo_cmap_binary if is_binary else geo_cmap_multi))
    plt.tight_layout()
    plt.show()
    
    #%% Writing data to the local drive
    
    output_file_name = getSaveFileName('Enter the name of the classified raster:', initialDir='your_initial_directory_here', fileTypes=[('TIF files', '*.tif')])
    # If the user hits cancel on saving the file, exit the function
    if outputFileName == '':
        print('Skipping saving the results for case: ' + caseID)
        return
    suffix = '_binary' if isBinary else ''
    outputFileName = ''.join([outputFileName.rstrip('.tif'), suffix, '.tif'])
    classifiedRaster = s2Imgs[0].copy(data=classifiedImg_forRaster)
    classifiedRaster = classifiedRaster.assign_attrs({'long_name': 'LCCImg'})
    classifiedRaster.rio.set_nodata(251)
    classifiedRaster.rio.to_raster(outputFileName)  
            
#%% Main function

def main():
    """
    Main function. Runs a while loop to process S-1 images and generate features until the user exits by pressing 0 on a prompt.

    Returns
    -------
    None.

    """
    proceed = True
    while proceed:
        plt.close('all')
        caseID = input('Enter a label identifying the case to classify: ')
        classType = input('Enter 1 for multi-class, enter 2 for water/not-water binary classification: ')
        if classType == '1':
            isBinary = False
        elif classType == '2':
            isBinary = True
        else:
            raise ValueError('Enter 1 or 2.')
        s1s2Classify(caseID, isBinary)
        proceed = int(input("Enter 0 to exit, any other number to continue processing images: "))
    

#%% Run main function

if __name__ == '__main__':
    ## Create a custom colormap for visualizing classification
    geo_cmap_multi = LinearSegmentedColormap.from_list('GEO', ['aqua', 'darkred', 'goldenrod', 'forestgreen'])
    geo_cmap_binary = LinearSegmentedColormap.from_list('GEO', ['darkred', 'aqua'])
    main()