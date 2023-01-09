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
# create an instance of the dataset* 

# use connectivity model

# regress cerebellar data onto cortical data 
    # get residuals

# get summary (a wrapper)

# set the directory of your dataset here:
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'

data_dir = base_dir + '/WMFS'
atlas_dir = base_dir + '/Atlases'

# create an instance of the dataset
Dat = DataSetWMFS(data_dir)

# 1. run this case if you have not extracted data for the atlas
def extract_suit(dataSet, ses_id, type, atlas):
    # create an instance of the dataset class
    dataset = dataSet(data_dir)

    # extract data for suit atlas
    dataset.extract_all_suit(ses_id,type,atlas)

    return

# 2. run this case if you have not extracted data for the atlas    
def extract_fs32K(dataSet, ses_id, type):
    # create an instance of the dataset class
    dataset = dataSet(data_dir)

    # extract data for suit atlas
    dataset.extract_all_fs32k(ses_id,type)
    return

# 3. aggregate data for each parcel within the parcellation
def agg_data_parcel(data, info, atlas, label_img):
    """
    aggregate data over the voxels/vertices within a parcel.
    Args:
    data (np.ndarray)
    atlas (Atlas obj)
    label_img (str or list of str)

    Returns:
    data_parcel (np.ndarray)     
    """

    # get parcel and parcel axis using the label image
    atlas.get_parcel(label_img=label_img)
    atlas.get_parcel_axis()

    if len(atlas.label_vector) == 1: # for cerebellum (not divided by hemi)
        vector = atlas.label_vector[0]
    else: # for cortex, divided into left and right
        vector = np.concatenate(atlas.label_vector, axis = 0)

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
def predict_cerebellum():
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
def get_summary_whole():
    """
    prepares a dataframe for plotting the scatterplot with X and Y averaged over the whole cerebellum and cortex
    """

    # get participants for the dataset
    T = Dat.get_participants()

    # getting data for all the participants in SUIT space
    cdat, info = Dat.get_data(space='SUIT3', ses_id='ses-02', type='CondHalf', fields=None)
    # getting data for all the participants in fs32k space
    ccdat, info = Dat.get_data(space='fs32k', ses_id='ses-02', type='CondHalf', fields=None)

    # average over the whole structures
    Y = np.nanmean(cdat, axis = 2)
    X = np.nanmean(ccdat, axis = 2)

    # looping through subjects and doing regression for each subject
    summary_list = []
    for sub in range(len(T.participant_id)):
        info_sub = info.copy()
        print(f"- getting info for {T.participant_id[sub]}")
        x = X[sub, :]
        y = Y[sub, :]

        coef, res, R2 = regressXY(x, y, subtract_mean = False)
        
        info_sub["sn"] = T.participant_id[sub]
        info_sub["X"] = x
        info_sub["Y"] = y
        info_sub["res"] = res
        info_sub["coef"] = coef * np.ones([len(info), 1])
        info_sub["R2"] = R2 * np.ones([len(info), 1])

        summary_list.append(info_sub)
    
    summary_df = pd.concat(summary_list, axis = 0)
    return summary_df

# 6. getting data into a dataframe
def get_summary_roi():
    """
    prepares a dataframe for plotting the scatterplot
    """
    return




if __name__ == "__main__":
    df = get_summary_whole()
    # save the dataframe for later
    filepath = os.path.join(data_dir, 'sc_df_whole_ses-02.tsv')
    df.to_csv(filepath, index=False,sep='\t')

