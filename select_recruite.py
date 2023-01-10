#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions of atlas definition and atlas mapping

Created on 12/09/2022 at 5:25 PM
Author: Ladan Shahshahani
"""
# import packages
import sys
sys.path.append('../Functional_Fusion') 
sys.path.append('../cortico-cereb_connectivity') 

import numpy as np
import pandas as pd
from pathlib import Path

# modules from functional fusion
from atlas_map import *
from dataset import *
from matrix import indicator

import os
import nibabel as nb



# WHAT TO DO?


# use connectivity model


# set the directory of your dataset here:
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'

# Creating an instance of Dataset object for your dataset
def get_class(dataset_name = "WMFS"):
    """
    gets dataset class for your dataset of interest
    Args:
    dataset_name (str) - str representing the name assigned to the dataset in FunctionalFusion folder
    Returns:
    mydataset (DataSet object) - object representing the class for the dataset
    """
    # get the dataset directory and class
    T = pd.read_csv(base_dir + '/dataset_description.tsv',sep='\t')
    T.name = [n.casefold() for n in T.name]
    i = np.where(dataset_name.casefold() == T.name)[0]
    if len(i)==0:
        raise(NameError(f'Unknown dataset: {dataset_name}'))
    dsclass = getattr(sys.modules[__name__],T.class_name[int(i)])
    dir_name = base_dir + '/' + T.dir_name[int(i)]
    mydataset = dsclass(dir_name)
    return mydataset

# 1. run this case if you have not extracted data for the atlas
def extract_suit(dataset_name, ses_id, type, atlas):
    # create an instance of the dataset class
    dataset = get_class(dataset_name)

    # extract data for suit atlas
    dataset.extract_all_suit(ses_id,type,atlas)

    return

# 2. run this case if you have not extracted data for the atlas    
def extract_fs32K(dataset_name, ses_id, type):
    # create an instance of the dataset class
    dataset = get_class(dataset_name)

    # extract data for suit atlas
    dataset.extract_all_fs32k(ses_id,type)
    return

# 3. aggregate data for each parcel within the parcellation
def agg_data_parcel(data, info, atlas, label_img, unite_hemi = True):
    """
    aggregate data over the voxels/vertices within a parcel.
    Args:
    data (np.ndarray) - data to be aggregated
    atlas (Atlas obj) - an instance of the atlas object
    label_img (list)  - str or list of str representing the label images
    mask_img (list)   - list of mask str representing paths to the mask used when creating the data

    Returns:
    data_parcel (np.ndarray) - numpy array #condition-by-#parcels      
    """
    # get parcel and parcel axis using the label image
    atlas.get_parcel(label_img=label_img, unite_hemi = unite_hemi)
    atlas.get_parcel_axis()

    if len(atlas.label_vector) == 1: # for cerebellum (not divided by hemi)
        vector = atlas.label_vector[0]

    else: # for cortex, divided into left and right
        vector = np.concatenate(atlas.label_vector, axis = 0)
        # get the mask used to create the atlas
        mask = np.concatenate(atlas.mask, axis = 0)
        # apply the mask
        # NOTE: this could be deleted if we apply the left and right hemi mask to 
        # all the surface label giftis we have under atl-fs32k
        vector = vector[mask > 0]    
    
    # create a matrix for aggregating data (cannot use dataset.agg_data now! Need to make changes)
    C = indicator(vector,positive=True)

    # get the mean across parcel
    data = np.nan_to_num(data) # replace nans first 
    data_parcel = (data @ C)/np.sum(C, axis = 0)

    # create a dscale cifti with parcelAxis labels and data_parcel
    row_axis = info.cond_name
    row_axis = nb.cifti2.ScalarAxis(row_axis)

    parcel_axis = atlas.parcel_axis
    # HEAD = cifti2.Cifti2Header.from_axes((row_axis,bm,pa))
    header = nb.Cifti2Header.from_axes((row_axis, parcel_axis))
    cifti_img = nb.Cifti2Image(dataobj=data_parcel, header=header)
    
    return cifti_img

# 4. OPTIONAL step: use connectivity model to predict cerebellar activation
def predict_cerebellum(X, w):
    """
    makes predictions for the cerebellar activation
    uses weights from a linear model (w) and cortical data (X)
    to make predictions Yhat
    Args:
    X (np.ndarray)
    w (np.ndarray)
    Returns:
    Yhat (np.ndarray)
    """
    return

# 5. regress cerebellar data onto step 4 (or 4:alternative)
def regressXY(X, Y, subtract_mean = False):
    """
    regresses Y onto X.
    Will be used to regress observed cerebellar data onto predicted
    Args:
        X (np.ndarray) - predicted cerebellar data for each roi
        Y (np.ndarray) - observed cerebellar data for each roi
        subtract_mean (boolean) - subtract mean before regression?
    Returns:
        coef (np.ndarray) - regression coefficients
        residual (np.ndarray) - residuals 
        R2 (float) - R2 of the regression fit
    """

    # subtract means?
    if subtract_mean:
        X = X - X.mean(axis = 0, keepdims = True)
        Y = Y - Y.mean(axis = 0, keepdims = True)

    # Estimate regression coefficients
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    coef = np.linalg.inv(X.T@X) @ (X.T@Y)
    
    # matrix-wise simple regression?????????????????????????
    # c = np.sum(X*Y, axis = 0) / np.sum(X*X, axis = 0)


    # Calculate residuals
    residual = Y - X@coef

    # calculate R2 
    rss = sum(residual**2)
    tss = sum((Y - np.mean(Y))**2)
    R2 = 1 - rss/tss

    return coef, residual, R2

# 6. getting data into a dataframe
def get_summary(dataset_name = "WMFS",
                cerebellum = 'Buckner7', 
                cortex = 'yeo7', 
                agg_whole = False):
    """
    prepares a dataframe for plotting the scatterplot
    """
    # get the dataset class
    Dat = get_class(dataset_name= dataset_name)
    # get participants for the dataset
    T = Dat.get_participants()

    # create instances of atlases for the cerebellum and cortex
    mask_cereb = atlas_dir + '/tpl-SUIT' + '/tpl-SUIT_res-3_gmcmask.nii'
    atlas_cereb = AtlasVolumetric('SUIT3', mask_img=mask_cereb, structure="cerebellum")

    mask_cortex = []
    for hemi in ['L', 'R']:
        mask_cortex.append(atlas_dir + '/tpl-fs32k' + f'/tpl-fs32k_hemi-{hemi}_mask.label.gii')
    name = 'fs32k'
    atlas_cortex = AtlasSurface(name, mask_img=mask_cortex, structure=["cortex_left", 'cortex_right'])

    # get label images for the cerebellum and cortex
    if agg_whole: # using masks as labels
        # using 1 label over the whole structure (mask = 0 where there's no data mask = 1 otherwise)
        label_cereb = mask_cereb
        label_cortex = mask_cortex
    else:

        label_cereb = atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum}_space-SUIT_dseg.nii'

        label_cortex = []
        for hemi in ['L', 'R']:
            label_cortex.append(atlas_dir + '/tpl-fs32k' + f'/{cortex}.{hemi}.label.gii')


    # getting data for all the participants in SUIT space
    cdat, info = Dat.get_data(space='SUIT3', ses_id='ses-02', type='CondHalf', fields=None)
    # getting data for all the participants in fs32k space
    ccdat, info = Dat.get_data(space='fs32k', ses_id='ses-02', type='CondHalf', fields=None)

    # loop through subjects and create a dataframe
    summary_list = []
    for sub in range(len(T.participant_id)):
        
        # get data for the current subject
        this_data_cereb = cdat[sub, :, :]

        this_data_cortex = ccdat[sub, :, :]

        # pass on the data with the atlas object to the aggregating function
        cifti_Y = agg_data_parcel(this_data_cereb, info, atlas_cereb, label_cereb, unite_hemi=False)
        cifti_X = agg_data_parcel(this_data_cortex, info, atlas_cortex, label_cortex, unite_hemi=True)

        # get data per parcel
        X = cifti_X.get_fdata()
        Y = cifti_Y.get_fdata()

        # use connectivity model to make predictions
        # w, scale = get_connectivity() # getting the connectivity weights and scaling factor
        # Yhat  = predict_cerebellum(X, w, scale)
        # X = Yhat.copy()

        # looping over labels and doing regression for each corresponding label
        for ilabel in range(Y.shape[1]):
            info_sub = info.copy()
            print(f"- subject {T.participant_id[sub]} label {ilabel+1}")
            x = X[:, ilabel]
            y = Y[:, ilabel]

            coef, res, R2 = regressXY(x, y, subtract_mean = False)

            info_sub["sn"] = T.participant_id[sub]
            info_sub["X"] = x
            info_sub["Y"] = y
            info_sub["res"] = res
            info_sub["coef"] = coef * np.ones([len(info), 1])
            info_sub["R2"] = R2 * np.ones([len(info), 1])
            info_sub['label'] = (ilabel+1) * np.ones([len(info), 1])

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0) 
    return summary_df


if __name__ == "__main__":

    """
    Getting the summary dataframe for the scatterplot in an ROI-wise manner
    """
    df = get_summary(dataset_name = "WMFS", agg_whole=False)
    # save the dataframe for later
    filepath = os.path.join(base_dir, 'WMFS', 'sc_df_yeo_buckner_7_ses-02.tsv')
    df.to_csv(filepath, index = False, sep='\t')

    """
    Getting the summary dataframe for the scatterplot over whole structures
    """
    df = get_summary(dataset_name = "WMFS", agg_whole=True)
    # save the dataframe for later
    filepath = os.path.join(base_dir, 'WMFS', 'sc_df_whole_ses-02.tsv')
    df.to_csv(filepath, index = False, sep='\t')

