#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to do selective recruitment analysis

Created on 12/09/2022 at 5:25 PM
Author: Ladan Shahshahani
"""
# import packages
import numpy as np
import pandas as pd
from pathlib import Path

# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.matrix as matrix

# modules from connectivity
# import cortico_cereb_connectivity.data as cdata

import nibabel as nb
import nitools as nt


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

def calc_mean(data,info,
                partition='run',
                condition='reg_id',
                reorder=False):

    """
    Calculating mean per condition
    Args:
        data 
        partition
        condition
        reorder
    Returns:
        mean_d
        info
    """
    n_subj = data.shape[0]
    cond=np.unique(info.reg_id)
    n_cond = len(cond)

    # For this purpose, ignore nan voxels
    data = np.nan_to_num(data,copy=False)
    
    mean_d = data.mean(axis=2)
    Z = matrix.indicator(info.reg_id)
    mean_d = mean_d @ np.linalg.pinv(Z).T
    
    part = np.unique(info[partition])
    inf=info[info[partition]==part[0]].copy()
    if reorder:
        inf=inf.sort_values(reorder)
        ind=inf.index.to_numpy()
        inf=inf.reset_index()
        mean_d = mean_d[:,ind]
    
    return mean_d,inf

# use connectivity model to predict cerebellar activation
def predict_cerebellum(weights, scale, X, atlas, info, fwhm = 0):
    """
    makes predictions for the cerebellar activation
    uses weights from a linear model (w) and cortical data (X)
    to make predictions Yhat
    Args:
    X (np.ndarray)       - cortical data
    weights (np.ndarray) - connectivity weights
    scale (np.ndarray)   - used to scale data
    atlas (atlas object) - atlas object (will be used in smoothing)
    info (pd.DataFrame)  - pandas dataframe representing task info
    fwhm (int)           - smoothing
    Returns:
    Yhat (np.ndarray) - predicted cerebellar data
    """

    # replace Nans
    X = np.nan_to_num(X)
    # Y = np.nan_to_num(Y)  

    # apply scaling
    X = X / scale

    # get smoothing matrix 
    if fwhm != 0:
        smooth_mat = get_smooth_matrix(atlas, fwhm)
        weights = smooth_mat@weights

    # loop over subjects
    n_subj,n_cond, n_parcel = X.shape
    Yhat = np.zeros([n_subj, n_cond, weights.shape[0]])
    for i in range(n_subj):
        x = X[i,:, :]
        # make predictions
        Yhat[i, :, :] = np.dot(x, weights.T)
        Yhat[i, :, :] = np.r_[Yhat[i, info.half == 2, :], Yhat[i, info.half == 1, :]]
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
    if fit_intercept:
        X = np.c_[ np.ones(X.shape[0]), X ]  

    coef = np.linalg.pinv(X) @ Y 
    predict = X @ coef
    residual = Y - predict
    # print(sum(residual))

    # calculate R2 
    rss = sum(residual**2)
    tss = sum((Y - np.mean(Y))**2)
    R2 = 1 - rss/tss

    return coef[-1], residual, R2

def run_regress(X,Y,info,fit_intercept = False):
    """ Runs regression analysis for each subject and ROI. 
    Args:
        X (ndarray): n_subj x n_cond x n_roi
        Y (ndarray): n_subj x n_cond x n_roi
        info (DataFrame): information on conditions 
        fit_intercept (bool): Use intercept in regression. Default = False
    Returns:
        df (DataFrame): resulting data frame
    """
    n_subj,n_cond,n_roi = X.shape

    summary_list = [] 
    for i in range(n_subj):
        for r in range(n_roi):
            info_sub = info.copy()
            x = X[i,:,r]
            y = Y[i,:,r]

            coef, res, R2 = regressXY(x, y, fit_intercept = fit_intercept)
            vec = np.ones((len(info_sub),))
            info_sub["sn"]    = i * vec
            info_sub["roi"]   = r * vec
            info_sub["X"]     = x # X is either the cortical data or the predicted cerebellar activation
            info_sub["Y"]     = y
            info_sub["res"]   = res
            info_sub["coef"]  = coef * vec
            info_sub["R2"]    = R2 * vec

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0)
    return summary_df

# make cerebellar roi
def make_roi_cerebellum(cifti_img, info, threshold, atlas_space = "SUIT3", localizer = "Verbal2Back"):
    """
    creates label nifti for roi cerebellum
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    localizer (str) - name of the localizer
    Returns:
    nifti_img (nb.nifti) - nifti of the label created
    """
    # get suit data
    data = cifti_img.get_fdata()
    # get the row index corresponding to the contrast
    info_con = info.loc[info.cond_name == localizer]
    # get the map for the contrast of interest
    con_map  = data[info_con.cond_num.values -1, :]

    # get threshold value (ignoring nans)
    percentile_value = np.nanpercentile(con_map, q=threshold)

    # apply threshold
    thresh_data = con_map > percentile_value
    # convert 0 to nan
    thresh_data[thresh_data != False] = np.nan

    # create an instance of the atlas (will be used to convert data to nifti)
    atlas, a_info = am.get_atlas(atlas_space,atlas_dir)
    nifti_img = atlas.data_to_nifti(1*thresh_data)
    return nifti_img

# make cortical roi
def make_roi_cortex(cifti_img, info, threshold, localizer = "Verbal2Back"):
    """
    creates label giftis for left and right hemisphere of the cortex
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    localizer (str) - name of the localizer
    Returns:
    gifti_img (list) - list of label giftis for left and right hemispheres
    """
    # get data for left and right hemisphere
    data_list = nt.surf_from_cifti(cifti_img)

    # threshold and create label
    gifti_img = []
    for i, name in zip([0, 1], ['CortexLeft', 'CortexRight']):
        # get data for the hemisphere
        data = data_list[i]

        # get the contrast map
        # get the row index corresponding to the contrast
        info_con = info.loc[info.cond_name == localizer]
        # get the map for the contrast of interest
        con_map  = data[info_con.cond_num.values -1, :]

        # get threshold value (ignoring nans)
        percentile_value = np.nanpercentile(con_map, q=threshold)

        # apply threshold
        thresh_data = con_map > percentile_value
        # convert 0 to nan
        thresh_data[thresh_data != False] = np.nan
        # create label gifti
        gifti_img.append(nt.make_label_gifti(1*thresh_data.T, anatomical_struct=name))
    return gifti_img