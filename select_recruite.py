#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to do selective recruitment analysis

Created on 12/09/2022 at 5:25 PM
Author: Ladan Shahshahani
"""
# import packages
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

import os
import nibabel as nb
import nitools as nt


# set base directory of the functional fusion 
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'

# functions used to create nifti/gifti labels for the region of interest
def extract_group_data(dataset_name = "MDTB"):
    """
    """
    # get the Dataset class
    Data = get_class(dataset_name=dataset_name)
    
    # get group average. will be saved under <dataset_name>/derivatives/group
    Data.group_average_data(ses_id="ses-s2",
                                 type="CondAll",
                                 atlas='SUIT3')

    Data.group_average_data(ses_id="ses-s2",
                                 type="CondAll",
                                 atlas='fs32k')
    return

def make_roi_cerebellum(cifti_img, info, threshold, atlas_space = "SUIT3", contrast = "Verbal2Back"):
    """
    creates label nifti for roi cerebellum
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    contrast (str) - name of the contrast
    Returns:
    nifti_img (nb.nifti) - nifti of the label created
    """
    # get suit data
    data = cifti_img.get_fdata()
    # get the row index corresponding to the contrast
    info_con = info.loc[info.cond_name == contrast]
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

def make_roi_cortex(cifti_img, info, threshold, contrast = "Verbal2Back"):
    """
    creates label giftis for left and right hemisphere of the cortex
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    contrast (str) - name of the contrast
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
        info_con = info.loc[info.cond_name == contrast]
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

def make_roi(dataset_name = "MDTB", 
                     contrast = "Verbal2Back", 
                     ses_id = "ses-s1", 
                     threshold = 80):
    """
    Uses a group contrast to create label files (nifti and gifti)
    
    Args:
    dataset_name (str) - dataset
    contrast (str) - name of the contrast (see info tsv file) to be used for roi creation
    ses_id (str) - session id where the contrast of interest is
    threshold (int) - percentile value (1-100)

    """
    # get Dataset class for your dataset
    Data = get_class(dataset_name=dataset_name)

    # load data 
    cifti_cerebellum = nb.load(Data.data_dir.format("group") + f"/group_space-SUIT3_{ses_id}_CondAll.dscalar.nii")
    cifti_cortex = nb.load(Data.data_dir.format("group") + f"/group_space-fs32k_{ses_id}_CondAll.dscalar.nii")
    
    # load info (will be used to select contrast)
    info_tsv = pd.read_csv(Data.data_dir.format("group") + f"/group_ses-s1_info-CondAll.tsv", sep="\t")

    # label files for the cerebellum and cortex
    roi_nifti = make_roi_cerebellum(cifti_cerebellum, info_tsv, threshold, atlas_space = "SUIT3", contrast = "Verbal2Back")
    roi_gifti = make_roi_cortex(cifti_cortex, info_tsv, threshold, contrast = "Verbal2Back")

    # save label files
    nb.save(roi_nifti, Data.atlas_dir + '/tpl-SUIT' + f'/atl-{contrast}_space-SUIT_dseg.nii')

    for i, h in enumerate(['L', 'R']):
        nb.save(roi_gifti[i], Data.atlas_dir + '/tpl-fs32k' + f'/{contrast}.32k.{h}.label.gii')

    return


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

# Get smoothing matrix
def get_smooth_mat(atlas, fwhm = 3):
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

def get_parcels_remove(ntessels):
    """
    ONLY USE THIS BEFORE RE_RUNNING THE CONNECTIVITY
    IT GIVES YOU THE TESSELS/PARCELS THAT ARE TO BE EXCLUDED
    """

    # get the label file for the tesselation
    tessel_dir = '/srv/diedrichsen/data/Cerebellum/super_cerebellum/sc1/RegionOfInterest/data/group'

    tessel_file = []
    for h in ['L', 'R']:
        tessel_file.append(nb.load(os.path.join(tessel_dir, f'tessels{ntessels:04d}.{h}.label.gii')))

    # get the mask used in atlas
    atlas_cortex, ainfo = am.get_atlas('fs32k', atlas_dir)

    aa = []
    xx = []
    for i in [0, 1]:
        x = tessel_file[i].agg_data()
        y = atlas_cortex.vertex[i]
        xy =x[y]
        aa.append(xy)
        xx.append(x)
    aa = np.concatenate(aa)
    xx = np.concatenate(xx)
    
    # find regions that are in xx but not in aa
    exclude_r = np.unique(xx[~np.isin(xx,aa)])

    return exclude_r
# 1. run this case if you have not extracted data for the atlas
def extract_data(dataset_name, ses_id, type, atlas):
    # create an instance of the dataset class
    dataset = get_class(dataset_name)

    # extract data for suit atlas
    dataset.extract_all(ses_id,type,atlas)

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
def predict_cerebellum(X, atlas, info, method = 'ridge', fwhm = 3):
    """
    makes predictions for the cerebellar activation
    uses weights from a linear model (w) and cortical data (X)
    to make predictions Yhat
    Args:
    X (np.ndarray)      - cortical data
    path2conn (str)     - path to where connectivity scaling factor and weights are stored
    method (str)        - method used for estimating connectivity weights. Default is Ridge regression
    smooth (np.ndarray) - smoothing kernel. if None, data will not be smoothed
    Returns:
    Yhat (np.ndarray) - predicted cerebellar data
    """
    # load in the csv file containing the information for best models
    model_list = pd.read_csv(os.path.join(base_dir, "WMFS", "conn", f'best_models_{method}.csv'), sep = ',')

    # get name of the cortical tesseelation used
    n_tessels = X.shape[1]
    cortex_name = f"tessels{n_tessels/2:02d}"

    # get the model_name
    mymodel = model_list.loc[model_list["cortex_names"] == cortex_name]
    model_name = mymodel["models"].values[0]
    model_path = os.path.join(base_dir, "WMFS", "conn", f"{model_name}.h5")
    scale_path = os.path.join(base_dir, "WMFS", "conn", f"{cortex_name}_scale.h5")
    
    # load the model and scale
    model = dd.io.load(model_path)
    scale = dd.io.load(scale_path)

    W = model['weights'].copy()
    Xs = X / scale

    # get smoothing matrix 
    if fwhm != 0:
        smooth_mat = get_smooth_mat(atlas, fwhm)
        W = smooth_mat@W

    # make predictions
    Yhat = np.dot(Xs, W.T)
    Yhat = np.r_[Yhat[info.half == 2, :], Yhat[info.sess == 1, :]]


    return Yhat

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
                cortex = 'Icosahedron-1002.32k', 
                agg_whole = False):
    """
    prepares a dataframe for plotting the scatterplot
    """
    # get the dataset class
    Dat = get_class(dataset_name= dataset_name)
    # get participants for the dataset
    T = Dat.get_participants()

    # create instances of atlases for the cerebellum and cortex
    atlas_cereb, ainfo = am.get_atlas('SUIT3',atlas_dir)
    mask_cereb = atlas_dir + '/tpl-SUIT' + '/tpl-SUIT_res-3_gmcmask.nii'
    # atlas_cereb = AtlasVolumetric('SUIT3', mask_img=mask_cereb, structure="cerebellum")

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
        # Yhat = predict_cerebellum(X, atlas_cereb, info, method = 'ridge', fwhm = 3) # getting the connectivity weights and scaling factor
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
    # df = get_summary(dataset_name = "WMFS", agg_whole=False)
    # # save the dataframe for later
    # filepath = os.path.join(base_dir, 'WMFS', 'sc_df_yeo_buckner_7_ses-02.tsv')
    # df.to_csv(filepath, index = False, sep='\t')

    """
    Getting the summary dataframe for the scatterplot over whole structures
    """
    # df = get_summary(dataset_name = "WMFS", agg_whole=True)
    # # save the dataframe for later
    # filepath = os.path.join(base_dir, 'WMFS', 'sc_df_whole_ses-02.tsv')
    # df.to_csv(filepath, index = False, sep='\t')

    """
    Make regions of interest from MDTB
    """
    # make_roi(dataset_name = "MDTB", threshold=90)

    """
    Getting the summary dataframe for the scatterplot in an ROI-wise manner
    """
    df = get_summary(dataset_name = "WMFS", agg_whole=False, cerebellum="Verbal2Back", cortex="Verbal2Back.32k")
    # save the dataframe for later
    filepath = os.path.join(base_dir, 'WMFS', 'sc_df_VWM_ses-02.tsv')
    df.to_csv(filepath, index = False, sep='\t')

