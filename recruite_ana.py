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

# https://www.humanconnectome.org/software/workbench-command/-all-commands-help
# use subprocess to smooth the cifti files
# -cifti-smoothing

def make_ant_post_label(atlas_space = "SUIT3"):
    """
    Create a mask that divides the cerebellum into anterior and posterior
    Args:
        atlas_space (str) - string representing the atlas space
    Returns:
        mask_data (np.ndarray)
        mask_nii (nb.NiftiImage)
    """
    # create an instance of atlas object
    atlas_suit, ainfo = am.get_atlas(atlas_space, gl.atlas_dir)

    # load in the lobules parcellation
    lobule_file = f"{gl.atlas_dir}/tpl-SUIT/atl-Anatom_space-SUIT_dseg.nii"
    
    # get the lobules in the atlas space
    lobule_data, lobules = atlas_suit.get_parcel(lobule_file)

    # load the lut file for the lobules 
    idx_lobule, _, lobule_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-Anatom.lut")

    # demarcate the horizontal fissure
    ## horizontal fissure is between crusI and crusII.
    ## everything above crusII is the anterior part
    ## find the indices for crusII
    crusII_idx = [lobule_names.index(name) for name in lobule_names if "CrusII" in name]
    posterior_idx = idx_lobule[min(crusII_idx):]
    anterior_idx = idx_lobule[0:min(crusII_idx)]

    # assign value 1 to the anterior part
    anterior_mask = np.isin(lobule_data, anterior_idx)

    # assign value 2 to the posterior part
    posterior_mask = np.isin(lobule_data, posterior_idx)
    return anterior_mask, posterior_mask

def get_smooth_matrix(atlas, fwhm = 3):
    """
    calculates the smoothing matrix to be applied to data
    Args:
        atlas - experiment name (fs or wm)
        fwhm (float)  - fwhm of smoothing kernel
    Rerurns:
        smooth_mat (np.ndarray) - smoothing matrix
    """    
    
    # calculate euclidean distance
    if hasattr(atlas, "vox"): # for cerebellum
        euc_dist = nt.euclidean_dist_sq(atlas.vox, atlas.vox)
        smooth_mat = np.exp(-1/2 * euc_dist/(fwhm**2))
        smooth_mat = smooth_mat /np.sum(smooth_mat, axis = 1);  
    elif hasattr(atlas, "vertex"): # for cortex
        # get smoothing matrix for each hemi
        smooth_mat = []
        for h in [0, 1]:
            euc_dist = nt.euclidean_dist_sq(atlas.vertex[h], atlas.vertex[h])
            s_mat_hemi = np.exp(-1/2 * euc_dist/(fwhm**2))
            s_mat_hemi = s_mat_hemi /np.sum(s_mat_hemi, axis = 1)
            smooth_mat.append(s_mat_hemi)
     

    return smooth_mat

def smooth_surface(cifti_file, 
                   surface_kernel=2.0,
                   volume_kernel = 1.0,  
                   smoothing_direction = 'ROW', 
                   prefix='s'):
    """_summary_

    Args:
        cifti_file (_str_): path to cifti file
        surface_kernel (float, optional): _smoothing kernel for surface_. Defaults to 2.0.
        volume_kernel (float, optional): _smoothing kernel for volume in cifti_. Defaults to 1.0.
        smoothing_direction (str, optional): _direction of smoothing_. Defaults to 'ROW'. other option: 'COLUMN'
        prefix (str, optional): _prefix added to smoothed file_. Defaults to 's'.
    """
    
    if not cifti_file:
        sys.exit('Error: No fileList of .func.gii files provided.')
    subprocess.run(["wb_command","-cifti-smoothing", cifti_file, surface_kernel, volume_kernel, smoothing_direction , f"{prefix}{cifti_file}"])

def get_reliability_summary(dataset = "WMFS", ses_id = "ses-02", subtract_mean = True):
    """
    Calculates cross-validated reliability with runs as cross validating folds
    Args:
        dataset (str) - name of the dataset
        ses_id (str) - id assigned to the session
        subtract_mean(bool) - subtract mean before calculating the reliability?
    Returns:
        df (pd.DataFrame) - summary dataframe containing cross validated reliability measure
    """
    
    # get the datasets
    Data = ds.get_dataset_class(gl.base_dir, dataset=dataset)

    # loop over cortex and cerebellum
    D = []
    for atlas in ["SUIT3", "fs32k"]:

        # get the data tensor
        tensor, info, _ = ds.get_dataset(gl.base_dir,dataset,atlas = atlas, sess=ses_id,type='CondRun', info_only=False)

        # loop over conditions and calculate reliability
        for c, _ in enumerate(info.cond_name.loc[info.run == 1]):
            # get condition info data
            part_vec = info.run
            cond_vec = (info.cond_num == c+1)*(c+1)

            # get all the info from info to append to the summary dataframe
            info_cond = info.loc[(info.cond_num == c+1) & (info.run == 1)]

            r = ds.reliability_within_subj(tensor, part_vec, cond_vec,
                                            voxel_wise=False,
                                            subtract_mean=subtract_mean)

            # prep the summary dataframe
            R = pd.DataFrame()
            R["sn"] = Data.get_participants().participant_id
            R["R"] = np.mean(r, axis = 1)
            R["atlas"] = [atlas]*(r.shape[0])
            R.reset_index(drop = True, inplace=True)

            # get the rest of the info
            R_info = pd.concat([info_cond]*r.shape[0], axis = 0)
            # drop sn column from R_info
            R_info = R_info.drop(labels="sn", axis = 1)
            R_info.reset_index(drop = True, inplace=True)
            R = pd.concat([R, R_info], axis = 1, join = 'outer', ignore_index = False)

            D.append(R)
    
    df = pd.concat(D, axis = 0)
    
    return df

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

def agg_data(tensor, atlas, label, unite_struct = True):
    """
    aggregates the data ready for regression over parcels or entire structure
    """
    # get data tensor
    atlas, ainfo = am.get_atlas(atlas,gl.atlas_dir)

    # NOTE: atlas.get_parcel takes in path to the label file, not an array
    if (isinstance(atlas, am.AtlasSurface)) | (isinstance(atlas, am.AtlasSurfaceSymmetric)):
        if label is not None:
            atlas.get_parcel(label, unite_struct = unite_struct)
            
        else: # passes on mask to get parcel if you want the average across the whole structure
            atlas.label_vector = np.ones((atlas.P,),dtype=int)

    else:
        if label is not None:
            atlas.get_parcel(label)
        else: # passes on mask to get parcel if you want the average across the whole structure
            atlas.label_vector = np.ones((atlas.P,),dtype=int)


    # aggregate over voxels/vertices within parcels
    data, parcel_labels = ds.agg_parcels(tensor , 
                                         atlas.label_vector, 
                                         fcn=np.nanmean)
    
    return data, ainfo, parcel_labels

def add_rest_to_data(X,Y,info):
    """ adds rest to X and Y data matrices and info
    Args:
        X (ndarray): n_subj,n_cond,n_reg data matrix  
        Y (ndarray):  n_subj,ncond,n_reg data matrix
        info (DataFrame): Dataframe with information 

    Returns:
        X (ndarray): n_subj,n_cond+1,n_reg data matrix  
        Y (ndarray):  n_subj,ncond+1,n_reg data matrix
        info (DataFrame): Dataframe with information 
    """
    n_subj,n_cond,n_reg = X.shape
    X_new = np.zeros((n_subj,n_cond+1,n_reg))
    X_new[:,:-1,:]=X
    Y_new = np.zeros((n_subj,n_cond+1,n_reg))
    Y_new[:,:-1,:]=Y
    a = pd.DataFrame({'cond_name':'rest',
                'reg_id':max(info.reg_id)+1,
                'cond_num':max(info.reg_id)+1},index=[0])
    info_new = pd.concat([info,pd.DataFrame(a)],ignore_index=True)
    return X_new,Y_new,info_new

def get_summary(dataset = "WMFS", 
                ses_id = 'ses-02', 
                type = "CondAll", 
                cerebellum_space = 'SUIT3',
                cortex_space = 'fs32k',
                cerebellum_roi =None,
                cortex_roi = None,
                add_rest = True, 
                divide_region = True):
    """
    get summary dataframe for cerebellum vs. cortex
    Args: 
        dataset_name (str) - name of the dataset (as is used in functional fusion framework)
        ses_id (str)       - name of the session in the dataset you want to use

    Returns:
        sum_df (pd.DataFrame) - summary dataframe
    """
    # Get the dataset from Functional Fusion framework
    tensor_cerebellum, info, _ = ds.get_dataset(gl.base_dir,
                                                dataset,
                                                atlas=cerebellum_space,
                                                sess=ses_id,
                                                type=type)
    tensor_cortex, info, _ = ds.get_dataset(gl.base_dir,
                                            dataset,
                                            atlas=cortex_space,
                                            sess=ses_id,
                                            type=type)
    
    # get label files for cerebellum and cortex
    ## if None is passed then it will be averaged over the whole
    if cerebellum_roi is not None:
        cerebellum_roi = gl.atlas_dir + '/' + cerebellum_roi + '_dseg.nii'
    if cortex_roi is not None:
        cortex_label = []
        # Ultimately replace this with label CIFTI, to avoid additional code-writing
        for hemi in ['L', 'R']:
            cortex_label.append(gl.atlas_dir + '/' + cortex_roi + f'.{hemi}.label.gii')
        cortex_roi = cortex_label

    # get the data for all the subjects for cerebellum
    Y_parcel, ainfo, cerebellum_parcel = agg_data(tensor_cerebellum, 
            atlas = cerebellum_space, 
            label = cerebellum_roi)

    X_parcel, ainfo, cortex_parcel = agg_data(tensor_cortex, 
            atlas = cortex_space, 
            label = cortex_roi)
    
    # Want to add rest as a condition?
    if add_rest:
        X_parcel,Y_parcel,info = add_rest_to_data(X_parcel,Y_parcel,info)
    
    # Transform into a dataframe with X and Y data 
    n_subj,n_cond,n_roi = X_parcel.shape

    summary_list = [] 
    for i in range(n_subj):
        for r in range(n_roi):
            info_sub = info.copy()
            vec = np.ones((len(info_sub),))
            info_sub["sn"]    = i * vec
            info_sub["roi"]   = r * vec
            info_sub["X"]     = X_parcel[i,:,r]
            info_sub["Y"]     = Y_parcel[i,:,r]

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0,ignore_index=True)
    return summary_df

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
        eig_val (np.ndarray) - eigen values sorted according to quadrant criterion
        eig_vec (np.ndarray) - eigen vectors sorted according to quadrant criterion
    """
    # Check that X and Y are one-dim
    
    # Subtract mean 
    if zero_mean:
        X = X - X.mean()
        Y = Y - Y.mean()

    # calculate covariance
    XX = np.c_[X, Y]
    cov = XX.T@XX

    # compute eigen vectors of the covariance
    eig_val, eig_vec = np.linalg.eig(cov)

    # sort eigen vectors by the general direction (not the magnitude of eigen value)
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
    # print(slope)

    # getting the intercept
    if zero_mean:
        intercept = slope * X.mean()
    else:
        intercept = 0

    # calculate residuals from PCA
    residual = XX @ eig_vec[:,1].T

    # get intercept and slope into coef list
    coef = [intercept,slope]
    return coef,residual, eig_val, eig_vec

def run_pca(df, zero_mean = False):
    """ Runs pca analysis for each subject and ROI. 
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

            coef,res, eig_val,eig_vec = pcaXY(df.X[indx].to_numpy(),
                                              df.Y[indx].to_numpy(), 
                                              zero_mean=zero_mean)
            vec = np.ones(res.shape)
            df.loc[indx,'res'] = res
            df.loc[indx,'slope'] = coef[1]*vec
            df.loc[indx, 'intercept'] = coef[0]*vec

            # calculate rss
            rss = eig_val[1]
            # df.loc[indx,'R2']= R2 * vec

    return df

def run_regress(df,fit_intercept = False):
    """ Runs regression analysis for each subject and ROI. 
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

def threshold_map(map_data, threshold, binarize = False):
    """applies threshold (in percent) to the data

    Args:
        data (np.ndarray): data to be thresholded
        threshold (fload): threshold (in percecnt) to be applied
    Returns:
        data_thresh (np.ndarray): thresholded data
    """
    # get threshold value (ignoring nans)
    percentile_value = np.nanpercentile(map_data, q=threshold)
    # apply threshold
    map_thresholded = np.where(map_data< percentile_value, map_data, 0)
    # map_thresholded = map_data >= percentile_value
    if binarize:
        map_thresholded = np.where(map_thresholded>=percentile_value, map_thresholded, 1)
    return map_thresholded

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
    atlas, a_info = am.get_atlas(atlas_space,gl.atlas_dir)
    nifti_img = atlas.data_to_nifti(1*thresh_data)
    return nifti_img

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

if __name__ == "__main__":
    get_reliability_summary(dataset = "WMFS", ses_id = "ses-02", subtract_mean = True)
    # D = get_summary(dataset = "WMFS", 
    #             ses_id = 'ses-02', 
    #             type = "CondAll", 
    #             cerebellum_roi =None, 
    #             cortex_roi = None,
    #             add_rest = True)

    # DD = run_pca(D, zero_mean = True)



    