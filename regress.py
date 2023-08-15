#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The common functions used to do selective recruitment analysis

Author: Ladan Shahshahani, Joern Diedrichsen
"""
# import packages
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import subprocess

# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.matrix as matrix
import selective_recruitment.globals as gl
# modules from connectivity
# import cortico_cereb_connectivity.data as cdata

import nibabel as nb
import nitools as nt
import warnings

def regressXY(X, Y, fit_intercept = False):
    """
    regresses Y onto X.
    Will be used to regress observed cerebellar data onto predicted
    Args:
        X (1d array) - predicted cerebellar data (or cortical data) for a single roi (vector)
        Y (1d array) - observed cerebellar data for a single roi (vector)
        subtract_mean (boolean) - subtract mean before regression?
    Returns:
        coef (np.ndarray) - regression coefficients
        residual (np.ndarray) - residuals
        R2 (float) - R2 of the regression fit
    """
    # Check that X and Y are one-dim
    if np.isnan(X+Y).any():
        coef = np.array([np.nan]*2)
        residual = np.zeros(X.shape)*np.nan
        R2 = np.nan
        return coef, residual, R2

    if fit_intercept:
        X = np.c_[ np.ones(X.shape[0]), X ]

    coef = np.linalg.pinv(X) @ Y
    predict = X @ coef
    residual = Y - predict

    # calculate R2
    rss = sum(residual**2)
    tss = sum((Y - np.mean(Y))**2)
    R2 = 1 - rss/tss

    return coef, residual, R2

def pcaXY(X, Y, zero_mean = False):
    """
    Applies PCA to X and Y
    Args:
        X (1d array) - predicted cerebellar data (or cortical data) for a single roi (vector)
        Y (1d array) - predicted cerebellar data (or cortical data) for a single roi (vector)
        zero_mean (bool) - zero center the data by subtracting the mean?
    Returns:
        coef (list) - list containing intercept and slope. coef[0] is the intercept
        residual (np.ndarray) - residuals as defined by the distance to the component
        comvar (float) - proportion of common variance (eigval[0]/sum(eigval))
        eig_vec (np.ndarray) - eigen vectors sorted according to quadrant criterion
    """
    # Check that X and Y are one-dim

    # Subtract mean
    if zero_mean:
        XX = np.c_[X-X.mean(), Y-Y.mean()]
    else:
        XX = np.c_[X, Y]

    # calculate covariance
    cov = XX.T@XX
    if (np.isnan(cov).any()) or ((cov==0).any()):
        coef = np.array([np.nan]*2)
        residual = np.zeros(X.shape)*np.nan
        comvar = np.nan
        return coef, residual, comvar, np.zeros((2,2))*np.nan

    # compute eigenvectors of the covariance
    eig_val, eig_vec = np.linalg.eig(cov)

    # sort eigenvectors by the general direction (not the magnitude of eigen value)
    # The first one must go into the ++ or -- quadrant
    # The second one into the -+ or +- quadrant
    # get the sign of the eigen vectors (will be used to determine the quadrant)
    vec_sign = np.sign(eig_vec)

    # determine the quadrant
    quad_type = vec_sign[0]*vec_sign[1]
    if quad_type[0]==-1:
        eig_vec=eig_vec[:,[1,0]]
        eig_val=[eig_val[1],eig_val[0]]
    elif quad_type[0]==0:
        raise(NameError('Crazy - X and Y are completely unrelated'))

    # Sort eigen vactors: Flip the eigenvectors in the right direction
    # Assuming that observed cerebellar is on yaxis and cortical/predicted cerebellar is on xaxis,
    # ++ and -+ are what we are interested in
    vec_sign = np.sign(eig_vec)
    eig_vec = eig_vec * vec_sign[1,:].reshape(1,2)

    # getting the slope
    slope = eig_vec[1,0]/eig_vec[0,0]

    # getting the intercept
    if zero_mean:
        intercept = Y.mean() - slope * X.mean()
    else:
        intercept = 0

    # calculate residuals from PCA
    residual = XX @ eig_vec[:,1].T

    # get intercept and slope into coef list
    coef = [intercept,slope]
    comvar = eig_val[0]/sum(eig_val)
    return coef,residual, comvar, eig_vec

def map_regress(X,Y,fit_intercept = True,fit = 'common'):
    """ Runs regression analysis for different subjects using a full map-wise approach
    Args:
        X (ndarray): Predicted cerebellar data (n_subjects x n_cond x n_voxels)
        Y (ndarray): Observed cerebellar data (n_subjects x n_cond x n_voxels)
        fit_intercept (bool): fir intercept in regression. Default = True
        fit (str): 'common' or 'separate' .
    Returns:
        res (np.array): Residuals (n_subjects x n_cond x n_voxels)
        coef (np.array): Coefficients (n_subjects x n_cond x 2)
        R2 (np.array): R2 (n_subjects) or (n_subjects x n_voxels)
    """
    n_subjs,n_cond,n_vox = X.shape

    res = np.zeros((n_subjs,n_cond,n_vox))
    if fit == 'common':
        R2 = np.zeros((n_subjs,))
        coef = np.zeros((n_subjs,2))
    else:
        R2 = np.zeros((n_subjs,n_vox))
        coef = np.zeros((n_subjs,n_vox,2))

    for s in range(n_subjs):
        if fit == 'common':
            good = np.logical_not(np.isnan(Y[s,:,:].sum(axis=0)))
            coef[s,:], r, R2[s] = regressXY(X[s][:,good].flatten(),
                                      Y[s][:,good].flatten(),
                                    fit_intercept = fit_intercept)
            res[s][:,good] = r.reshape(n_cond,-1)
        elif fit == 'separate':
            for v in range(n_vox):
                coef[s,v,:], res[s,:,v], R2[s,v] = regressXY(X[s,:,v],
                                            Y[s,:,v],
                                            fit_intercept = fit_intercept)
    return res,coef,R2

def map_pca(X,Y,zero_mean = False,fit = 'common'):
    """ Runs PCA analysis for different subjects using a full map-wise approach
    Args:
        X (ndarray): Predicted cerebellar data (n_subjects x n_cond x n_voxels)
        Y (ndarray): Observed cerebellar data (n_subjects x n_cond x n_voxels)
        zero_mean (bool): Remove the mean from X and Y. Default = False
        fit (str): 'common' or 'separate' .
    Returns:
        res (np.array): Residuals (n_subjects x n_cond x n_voxels)
        coef (np.array): Coefficients (n_subjects x n_cond x 2)
        R2 (np.array): R2 (n_subjects) or (n_subjects x n_voxels)
    """
    n_subjs,n_cond,n_vox = X.shape

    res = np.zeros((n_subjs,n_cond,n_vox))
    if fit == 'common':
        comvar = np.zeros((n_subjs,)) # Common variance
        coef = np.zeros((n_subjs,2))
    else:
        comvar = np.zeros((n_subjs,n_vox))
        coef = np.zeros((n_subjs,n_vox,2))

    for s in range(n_subjs):
        if fit == 'common':
            good = np.logical_not(np.isnan(Y[s,:,:].sum(axis=0)))
            coef[s,:], r, comvar[s],eigvec = pcaXY(X[s][:,good].flatten(),
                                      Y[s][:,good].flatten(),
                                    zero_mean = zero_mean)
            res[s][:,good] = r.reshape(n_cond,-1)
        elif fit == 'separate':
            for v in range(n_vox):
                coef[s,v,:], res[s,:,v], comvar[s,v],_ = pcaXY(X[s,:,v],
                                            Y[s,:,v],
                                            zero_mean = zero_mean)
    return res,coef,comvar

def roi_regress(df,fit_intercept = False):
    """ Runs regression analysis for each subject and ROI from a data frame
    Args:
        df (DataFrame): Data frame with sn, roi, X & Y (get_summary)
        fit_intercept (bool): Use intercept in regression. Default = False
    Returns:
        df (DataFrame): resulting data frame
    """
    subjs = np.unique(df.sn)
    rois = np.unique(df.roi)
    df['slope']=[0]*len(df)
    df['intercept']=[0]*len(df)
    df['R2']=[0]*len(df)
    df['res']=[0]*len(df)
    for s in subjs:
        for r in rois:
            indx = (df.sn==s) & (df.roi==r)

            coef, res, R2 = regressXY(df.X[indx].to_numpy(),
                                        df.Y[indx].to_numpy(),
                                        fit_intercept = fit_intercept)
            vec = np.ones(res.shape)
            df.loc[indx,'res'] = res
            df.loc[indx,'slope'] = coef[-1] * vec
            if fit_intercept:
                df.loc[indx,'intercept'] = coef[0] * vec
            df.loc[indx,'R2']= R2 * vec
    return df

def roi_pca(df, zero_mean = False):
    """ Runs pca analysis for each subject and ROI from a data frame
    Args:
        df (DataFrame): Data frame with sn, roi, X & Y (get_summary)
        zero_mean (bool): subtract mean from data before pca?. Default = False
    Returns:
        df (DataFrame): resulting data frame
    """
    subjs = np.unique(df.sn)
    rois = np.unique(df.roi)
    df['slope']=[0]*len(df)
    df['intercept']=[0]*len(df)
    df['R2']=[0]*len(df)
    df['res']=[0]*len(df)
    for s in subjs:
        for r in rois:
            indx = (df.sn==s) & (df.roi==r)

            coef,res, comvar,eig_vec = pcaXY(df.X[indx].to_numpy(),
                                               df.Y[indx].to_numpy(),
                                               zero_mean=zero_mean)
            vec = np.ones(res.shape)
            df.loc[indx,'res'] = res
            df.loc[indx,'slope'] = coef[1]*vec
            df.loc[indx, 'intercept'] = coef[0]*vec

            # calculate rss
            df.loc[indx,'comvar']= comvar * vec

    return df

if __name__ == "__main__":
    pass


