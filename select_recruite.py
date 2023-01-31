#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to do selective recruitment analysis

Created on 12/09/2022 at 5:25 PM
Author: Ladan Shahshahani
"""
# import packages
from pyexpat import model
import sys
sys.path.append('../Functional_Fusion') 
sys.path.append('../cortico-cereb_connectivity') 

import numpy as np
import pandas as pd
import deepdish as dd
from pathlib import Path

# modules from functional fusion
from atlas_map import *
from dataset import *
from matrix import indicator

# modules from connectivity
import prepare_data as prep

import os
import nibabel as nb
import nitools as nt

# set base directory of the functional fusion 
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'

# Get smoothing matrix, can be used to smooth the weights (for connectivity)
def get_smooth_matrix(atlas, fwhm = 3):
    """
    calculates the smoothing matrix to be applied to data
    Args:
        atlas - experiment name (fs or wm)
        fwhm (float)  - fwhm of smoothing kernel
    Rerurns:
        smooth_mat (np.ndarray) - smoothing matrix
    """
    # get voxel coordinates
    
    
    # calculate euclidean distance
    euc_dist = nt.euclidean_dist_sq(atlas.vox, atlas.vox)

    smooth_mat = np.exp(-1/2 * euc_dist/(fwhm**2))
    smooth_mat = smooth_mat /np.sum(smooth_mat, axis = 1);   

    return smooth_mat

# use connectivity model to predict cerebellar activation
def predict_cerebellum(weights, scale, X, atlas, info, fwhm = 0):
    """
    makes predictions for the cerebellar activation
    uses weights from a linear model (w) and cortical data (X)
    to make predictions Yhat
    Args:
    X (np.ndarray)      - cortical data
    weights (np.ndarray) - connectivity weights
    scale (np.ndarray) - used to scale data
    atlas (atlas object) - atlas object (will be used in smoothing)
    info (pd.DataFrame) - pandas dataframe representing task info
    fwhm (int) - smoothing
    Returns:
    Yhat (np.ndarray) - predicted cerebellar data
    """

    X = X / scale

    # get smoothing matrix 
    if fwhm != 0:
        smooth_mat = get_smooth_matrix(atlas, fwhm)
        weights = smooth_mat@weights

    # make predictions
    Yhat = np.dot(X, weights.T)
    Yhat = np.r_[Yhat[info.half == 2, :], Yhat[info.half == 1, :]]
    return Yhat

# regress cerebellar data onto cortical/cerebellar predictions
def regressXY(X, Y, fit_intercept = False):
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

    # Estimate regression coefficients
    # X = X.reshape(-1, 1)
    # Y = Y.reshape(-1, 1)
    if fit_intercept:
        X = np.c_[ np.ones(X.shape[0]), X ]  
    # coef = np.linalg.inv(X.T@X) @ (X.T@Y)
    # coef = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))

    # z,resid,rank,sigma = np.linalg.lstsq(X,Y)
    
    # # matrix-wise simple regression?
    # # c = np.sum(X*Y, axis = 0) / np.sum(X*X, axis = 0)

    # # Calculate residuals
    # residual = Y - X@coef
    # print(sum(residual))


    model = np.polyfit(X, Y, 1)
    predict = np.poly1d(model)
    residual = Y - predict(X)
    print(sum(residual))

    # calculate R2 
    rss = sum(residual**2)
    tss = sum((Y - np.mean(Y))**2)
    R2 = 1 - rss/tss

    return model[1], residual, R2

# getting data into a dataframe (roi-wise)
def get_summary_roi(outpath = None,
                dataset_name = "WMFS",
                cerebellum = 'Buckner7', 
                cortex = 'Icosahedron-1002.32k', 
                agg_whole = False,
                ses_id = 'ses-02',
                type = "CondHalf", 
                save_tensor = False):
    """
    prepares a dataframe for plotting the scatterplot
    """
    # get dataset class object
    Data = get_dataset_class(base_dir, dataset=dataset_name)

    # get info
    info = Data.get_info(ses_id,type)

    # get list of subjects:
    T = Data.get_participants()

    if save_tensor:
        # get data tensor for SUIT3
        prep.save_data_tensor(outpath=outpath,
                        dataset = dataset_name,
                        atlas='SUIT3',
                        sess=ses_id,
                        type=type)

        # get data tensor for fs32k
        prep.save_data_tensor(outpath=outpath,
                        dataset = dataset_name,
                        atlas='fs32k',
                        sess=ses_id,
                        type=type)

    # load data tensor for SUIT3
    file_suit = outpath + f'/{dataset_name}_SUIT3_{ses_id}_{type}.npy'
    cdat = np.load(file_suit)

    # load data tensor for fs32k
    file_fs32k = outpath + f'/{dataset_name}_fs32k_{ses_id}_{type}.npy'
    ccdat = np.load(file_fs32k)
    

    # create instances of atlases for the cerebellum and cortex
    atlas_cereb, ainfo = am.get_atlas('SUIT3',atlas_dir)
    mask_cereb = atlas_dir + '/tpl-SUIT' + '/tpl-SUIT_res-3_gmcmask.nii'

    mask_cortex = []
    for hemi in ['L', 'R']:
        mask_cortex.append(atlas_dir + '/tpl-fs32k' + f'/tpl-fs32k_hemi-{hemi}_mask.label.gii')
    atlas_cortex, ainfo = am.get_atlas('fs32k', atlas_dir)

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

    # get parcel for both atlases
    atlas_cereb.get_parcel(label_cereb)
    atlas_cortex.get_parcel(label_cortex, unite_struct = True)

    # loop through subjects and create a dataframe
    summary_list = []
    for sub in range(len(T.participant_id)):
        
        # get data for the current subject
        this_data_cereb = cdat[sub, :, :]

        this_data_cortex = ccdat[sub, :, :]

        # get mean across voxels within parcel
        Y = agg_parcels(this_data_cereb,atlas_cereb.label_vector,fcn=np.nanmean)
        X = agg_parcels(this_data_cortex,atlas_cortex.label_vector,fcn=np.nanmean)

        # get mean across halves
        X = np.nanmean(np.concatenate([X[info.half == 1, :], X[info.half == 2, :]], axis = 1), axis = 1)
        Y = np.nanmean(np.concatenate([Y[info.half == 1, :], Y[info.half == 2, :]], axis = 1), axis = 1)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        # looping over labels and doing regression for each corresponding label
        for ilabel in range(Y.shape[1]):
            info_sub = info.copy()
            info_sub = info_sub.loc[info_sub.half == 1]
            print(f"- subject {T.participant_id[sub]} label {ilabel+1}")
            x = X[:, ilabel]
            y = Y[:, ilabel]

            coef, res, R2 = regressXY(x, y, fit_intercept = False)

            info_sub["sn"] = T.participant_id[sub]
            info_sub["X"] = x
            info_sub["Y"] = y
            info_sub["res"] = res
            info_sub["coef"] = coef * np.ones([len(info_sub), 1])
            info_sub["R2"] = R2 * np.ones([len(info_sub), 1])
            info_sub['label'] = (ilabel+1) * np.ones([len(info_sub), 1])

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0) 
    return summary_df

# getting data into a dataframe (using connectivity)
def get_summary_conn(outpath = None,
                     dataset_name = "WMFS",
                     method = 'L2Regression',
                     cereb_roi = "wm_verbal", 
                     parcellation = 'Icosahedron-1002.32k', 
                     conn_dataset = 'MDTB',
                     conn_ses_id  = 'ses-s1',
                     conn_method = "L2Regression", 
                     log_alpha = 8, 
                     ses_id = 'ses-02',
                     type = "CondHalf", 
                     save_tensor = False):

    # get dataset class object
    Data = get_dataset_class(base_dir, dataset=dataset_name)

    # get info
    info = Data.get_info(ses_id,type)

    # get list of subjects:
    T = Data.get_participants()

    if save_tensor:
        # get data tensor for SUIT3
        prep.save_data_tensor(dataset = dataset_name,
                        atlas='SUIT3',
                        sess=ses_id,
                        type=type)

        # get data tensor for fs32k
        prep.save_data_tensor(dataset = dataset_name,
                        atlas='fs32k',
                        sess=ses_id,
                        type=type)



    # load data tensor for SUIT3
    file_suit = outpath + f'{dataset_name}_SUIT3_{ses_id}_{type}.npy'
    cdat = np.load(file_suit)

    # load data tensor for fs32k
    file_fs32k = outpath + f'{dataset_name}_fs32k_{ses_id}_{type}.npy'
    ccdat = np.load(file_fs32k)
    

    # create instances of atlases for the cerebellum and cortex
    atlas_cereb, ainfo = am.get_atlas('SUIT3',atlas_dir)
    atlas_cortex, ainfo = am.get_atlas('fs32k', atlas_dir)

    # get label files for cerebellum and cortex
    label_cereb = atlas_dir + '/tpl-SUIT' + f'/atl-{cereb_roi}_space-SUIT_dseg.nii'

    label_cortex = []
    for hemi in ['L', 'R']:
        label_cortex.append(atlas_dir + '/tpl-fs32k' + f'/{parcellation}.{hemi}.label.gii')

    # get parcel for both atlases
    atlas_cereb.get_parcel(label_cereb)
    atlas_cortex.get_parcel(label_cortex, unite_struct = False)

    # get dataset class for the connectivity training dataset
    path2file = os.path.join(prep.conn_dir, conn_dataset, "train")
   
    weights = np.load(os.path.join(path2file, f"{parcellation}_{conn_ses_id}_{conn_method}_logalpha_{log_alpha}_best_weights.npy"))
    scale = np.load(os.path.join(path2file, f'{conn_dataset}_scale.npy'))


    # loop through subjects and create a dataframe
    summary_list = []
    for sub in range(len(T.participant_id)):
        print(f"- Doing {sub}")
        # get data for the current subject
        this_data_cereb = cdat[sub, :, :]

        this_data_cortex = ccdat[sub, :, :]

        # pass on the data with the atlas object to the aggregating function
        # get mean across voxels within parcel
        Y_parcel = agg_parcels(this_data_cereb,atlas_cereb.label_vector,fcn=np.nanmean)
        X_parcel = agg_parcels(this_data_cortex,atlas_cortex.label_vector,fcn=np.nanmean)
        
        # get mean across halves
        X = np.nanmean(np.concatenate([X_parcel[info.half == 1, :], X_parcel[info.half == 2, :]], axis = 1), axis = 1)
        Y = np.nanmean(np.concatenate([Y_parcel[info.half == 1, :], Y_parcel[info.half == 2, :]], axis = 1), axis = 1)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        

        # use connectivity model to make predictions
        Yhat = predict_cerebellum(weights, scale, X_parcel, atlas_cereb, info, fwhm = 0) # getting the connectivity weights and scaling factor
        Yhat_parcel = agg_parcels(Yhat,atlas_cereb.label_vector,fcn=np.nanmean)
        XX = Yhat_parcel.copy()
        # average over halves
        XX = np.nanmean(np.concatenate([X_parcel[info.half == 1, :], X_parcel[info.half == 2, :]], axis = 1), axis = 1)
        XX = XX.reshape(-1, 1)

        # looping over labels and doing regression for each corresponding label
        for ilabel in range(Y_parcel.shape[1]):
            info_sub = info.copy()
            info_sub = info_sub.loc[info.half == 1]
            print(f"- subject {T.participant_id[sub]} label {ilabel+1}")
            x = XX[:, ilabel]
            y = Y[:, ilabel]

            coef, res, R2 = regressXY(x, y, fit_intercept = False)

            info_sub["sn"] = T.participant_id[sub]
            info_sub["X"] = x
            info_sub["Y"]    = y
            info_sub["res"]  = res
            info_sub["coef"] = coef * np.ones([len(info_sub), 1])
            info_sub["R2"] = R2 * np.ones([len(info_sub), 1])
            info_sub['label'] = (ilabel+1) * np.ones([len(info_sub), 1])

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0)
    return summary_df
