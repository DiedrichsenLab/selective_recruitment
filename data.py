#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to do selective recruitment analysis
Authors: Ladan Shahshahani, Joern Diedrichsen
"""

import os
import numpy as np
import deepdish as dd
import pandas as pd

from scipy import stats as sps # to calcualte confidence intervals, etc

import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am
import Functional_Fusion.matrix as matrix

import cortico_cereb_connectivity.globals as ccc_gl

import selective_recruitment.globals as gl
import selective_recruitment.regress as ra
import selective_recruitment.region as sroi

import nitools as nt

def add_rest_to_data(X,info):
    """ adds rest to X and Y data matrices and info
    Args:
        X (ndarray): n_subj,n_cond,n_reg data matrix
        info (DataFrame): Dataframe with information

    Returns:
        X (ndarray): n_subj,n_cond+1,n_reg data matrix
        info_new (DataFrame): Dataframe with information
    """
    n_subj,n_cond,n_reg = X.shape
    X_new = np.zeros((n_subj,n_cond+1,n_reg))
    X_new[:,:-1,:]=X
    a = pd.DataFrame({'cond_name':'rest',
                'reg_id':max(info.reg_id)+1,
                'cond_num':max(info.reg_id)+1},index=[0])
    info_new = pd.concat([info,pd.DataFrame(a)],ignore_index=True)
    return X_new,info_new

def get_voxdata_cereb_cortex(dataset = "MDTB",
                         ses_id = 'all',
                         subj = None,
                         cereb_space='SUIT3',
                         cortex_space = 'fs32k',
                         type = "CondAll",
                         add_rest = False):
    """ Gets the matching vertex / voxel data for cerebellum and cortex without applying a connectivity model - returned in atlas space


    Returns:
        Y (np.ndarray): observed cerebellar data (n_subj,n_cond,n_vox)
        X (np.ndarray): predicted cortical data (n_subj,n_cond,n_vox)
        info (pd.DataFrame): dataframe with info for data
    """

    # get observed cerebellar data
    Y,info,dset = ds.get_dataset(gl.base_dir,dataset,
                                    atlas=cereb_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)

    # get cortical data to be used as input to the connectivity model
    X,info,dset = ds.get_dataset(gl.base_dir,dataset,
                                    atlas=cortex_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)
    if add_rest:
        Y,_ = add_rest_to_data(Y,info)
        X,info = add_rest_to_data(X,info)
    return Y, X, info

def get_voxdata_obs_pred(dataset = "WMFS",
                         ses_id = 'ses-02',
                         subj = None,
                         atlas_space='SUIT3',
                         cortex = 'Icosahedron1002',
                         type = "CondAll",
                         add_rest = False,
                         mname_base = "MDTB_all_Icosahedron1002_L2Regression",
                         mname_ext = "_A8",
                         train_type = "train",
                         crossed = True):
    """gets the observed and predicted voxel data for a given dataset and model.
    If connectivity model set to None, observed cerebellar and cortical data are returned.

    Args:
        dataset (str):
            name of the dataset to make predictions for. Defaults to "WMFS".
        ses_id (str):
            string representing the session id. Defaults to 'ses-02'.
        subj (list or None):
            list of names of the subjects. Defaults to None: does for all subjects.
        atlas_space (str, optional):
            atlas space you want to have the data in. Defaults to 'SUIT3'.
        cortex (str, optional):
            name of the ROI definition to be used for the cortex. Defaults to 'Icosahedron1002'.
        type (str, optional):
            type of the data you want to use. Defaults to "CondAll": for averaged across all runs.
        add_rest (bool, optional):
            add fake rest (all zeros)? Defaults to False.
        mname_base (str, optional):
            model name base containing the modelling data set and session id. Defaults to "MDTB_all_Icosahedron1002_L2Regression".
            If set to None, no model is used.
        mname_est (str, optional):
            Extension for model. Defaults to "_A8".
        train_type (str, optional):
            Directory of where to fine the connectivity model. Defaults to "train".
        crossed (bool, optional):
            flip the order of halfs (type has to be CondHalf)? Defaults to True.

    Returns:
        Y (np.ndarray): observed cerebellar data (n_subj,n_cond,n_vox)
        YP (np.ndarray): predicted cerebellar data or cortical data (n_subj,n_cond,n_vox)
        atlas (Object): cortical atlas
        info (pd.DataFrame): dataframe with info for data
    """

    # get observed cerebellar data
    Y,info,dset = ds.get_dataset(gl.base_dir,dataset,
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)

    # get the connectivity model...
    if mname_base is not None:
        model_path = os.path.join(ccc_gl.conn_dir,atlas_space,train_type,mname_base)
        fname = model_path + f"/{mname_base}{mname_ext}_avg.h5" # get the model averaged over subjects
        json_name = model_path + f"/{mname_base}{mname_ext}_avg.json"
        conn_model = dd.io.load(fname)

    # get cortical data to be used as input to the connectivity model
    X,info,dset = ds.get_dataset(gl.base_dir,dataset,
                                    atlas="fs32k", # for cortex we always use fs32k
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)
    atlas,ainf = am.get_atlas('fs32k',gl.atlas_dir)
    # make strings for the cortical label files
    # If cortex parcellation is given, condense the cortex
    if cortex is not None:
        label=[gl.atlas_dir+'/tpl-fs32k/'+cortex+'.L.label.gii',
           gl.atlas_dir+'/tpl-fs32k/'+cortex+'.R.label.gii']
        atlas.get_parcel(label,unite_struct=False)

        X, _ = ds.agg_parcels(X, atlas.label_vector, fcn=np.nanmean)


    # If connectivity model is given, use it to get the cerebellar predicted data
    if mname_base is not None:
        if crossed: # flip the order of the halfs
            if type == "CondHalf":
                X  = np.concatenate([X[:, info.half == 2, :], X[:, info.half == 1, :]], axis=1)
            else:
                print("CondAll is chosen so no crossing!")
                pass
        YP = conn_model.predict(X)
    # If not, simply return the cortical data
    else:
        YP = X

    if add_rest:
        Y,_ = add_rest_to_data(Y,info)
        YP,info = add_rest_to_data(YP,info)
    return Y, YP, atlas, info

def average_rois(tensor,
                info,
                atlas_space = "SUIT3",
                atlas_roi = "NettekovenSym32",
                roi_selected = None,
                unite_struct = False,
                var = "Y"):
    """ Makes a summary dataframe for data averaged over voxels in a parcel
    Values will be stored in a column named var

    Args:
        tensor (np.ndarray):
            n_subj,n_cond,n_vox data matrix
        info (pd.DataFrame):
            dataframe containing task info
        atlas_space (str, optional):
            atlas_space. Defaults to "SUIT3".
        atlas_roi (str):
            Defaults to "NettekovenSym32".
        roi_selected (list, optional):
            List of the names of all ROIs to be used. Defaults to None: all ROIs are used.
        unite_struct (bool, optional):
            Unite ROIs across hemispheres. Defaults to False.
        var (str, optional):
            column name to be used in the summary dataframe. Defaults to "Y".

    Returns:
        summary_df (pd.Dataframe): a dataframe with the summary data for each parcecl within atlas_roi
    """
    atlas, ainfo = am.get_atlas(atlas_dir=gl.atlas_dir, atlas_str=atlas_space)
    # If atlas ROI is given, get it.
    if atlas_roi is not None:
        if (isinstance(atlas, am.AtlasSurface)) | (isinstance(atlas, am.AtlasSurfaceSymmetric)):
            labels = []
            for hemi in ['L', 'R']:
                labels.append(gl.atlas_dir + f'/{ainfo["dir"]}/{atlas_roi}.{hemi}.label.gii')
            label_vec, _ = atlas.get_parcel(labels, unite_struct = unite_struct)
        else:
            labels = gl.atlas_dir + f'/{ainfo["dir"]}/atl-{atlas_roi}_space-SUIT_dseg.nii'
            label_vec, _ = atlas.get_parcel(labels)

        # use lookuptable to get the names of the regions if lut file exists
        lutfile = f"{gl.atlas_dir}/{ainfo['dir']}/atl-{atlas_roi}.lut"
        if os.path.exists(lutfile):
            reg_id,_,reg_name = nt.read_lut(lutfile)
        else: # if lut file doesn't exist, just use the parcel labels
            reg_id = np.unique(label_vec)
            reg_name = [f"ROI_{i}" for i in reg_id]

        # get the data for each parcel
        label_vec,reg_id,reg_name = am.parcel_recombine(label_vec, roi_selected,reg_id,reg_name)
    else:
        label_vec = atlas.label_vector = np.ones((atlas.P,),dtype=int)
        reg_name =['0','average']

    parcel_data, parcel_labels = ds.agg_parcels(tensor ,
                                                label_vec,
                                                fcn=np.nanmean)

    n_subj, n_cond, n_roi = parcel_data.shape

    # make a summary dataframe
    summary_list = []
    for i in range(n_subj):
        for r in range(n_roi):
            info_sub = info.copy()
            vec = np.ones((len(info_sub),),dtype=int)
            info_sub["sn"]    = i * vec
            info_sub["roi"]   = parcel_labels[r] * vec
            info_sub["roi_name"] = reg_name[r+1]
            info_sub[var]     = parcel_data[i,:,r]

            summary_list.append(info_sub)

    summary_df = pd.concat(summary_list, axis = 0,ignore_index=True)

    return summary_df

def get_summary_roi(dataset = "WMFS",
                     ses_id = 'ses-02',
                     subj = None,
                     atlas_space = "SUIT3",
                     cerebellum_roi = "NettekovenSym32",
                     cerebellum_roi_selected = None,
                     cortex_roi = "Icosahedron1002",
                     cortex_roi_selected = None,
                     type = "CondHalf",
                     add_rest = True):

    """
    Function to get summary dataframe for ROI-based analysis of cerebellar data
    vs. the cortical data
    A list of selected matching cortical and cerebellar ROIs need to be given, otherwise
    the function will simply return the average data across the entire cortex and cerebellum
    """

    # get observed and predicted data for each cerebellar voxel
    Y, X, atlas, info = get_voxdata_obs_pred(dataset = dataset,
                                            ses_id = ses_id,
                                            subj = subj,
                                            atlas_space=atlas_space,
                                            type = type,
                                            cortex = None,
                                            mname_base = None,
                                            mname_ext  = None,
                                            add_rest = add_rest)

    # get the observed data averaged over region
    obs_df = average_rois(Y, info=info,
                              atlas_space = atlas_space,
                              atlas_roi = cerebellum_roi,
                              roi_selected=cerebellum_roi_selected,
                              unite_struct = False,
                              var = "Y")


    pred_df = average_rois(X, info=info,
                              atlas_space = 'fs32k',
                              atlas_roi = cortex_roi,
                              roi_selected=cortex_roi_selected,
                              unite_struct = False,
                              var = "X")

    # merge the dataframes
    obs_df['cortex_roi_name'] = pred_df['roi_name']
    obs_df['X'] = pred_df['X']

    return obs_df

def get_summary_conn(dataset = "MDTB",
                     ses_id = 'ses-01',
                     subj = None,
                     atlas_space = "SUIT3",
                     cerebellum_roi = "NettekovenSym32",
                     cerebellum_roi_selected = None,
                     cortex_roi = "Icosahedron1002",
                     type = "CondHalf",
                     add_rest = True,
                     mname_base = 'MDTB_all_Icosahedron_L2regression',
                     mname_ext = '_a8',
                     crossed = True):

    """
    Function to get summary dataframe for ROI-based analysis of cerebellar data
    vs. the predicted cerebellar data from a specific connectivty model.
    """

    # get observed and predicted data for each cerebellar voxel
    Y, Yhat, atlas, info = get_voxdata_obs_pred(dataset = dataset,
                                                ses_id = ses_id,
                                                subj = subj,
                                                atlas_space=atlas_space,
                                                cortex = cortex_roi,
                                                type = type,
                                                mname_base = mname_base,
                                                mname_ext  = mname_ext,
                                                add_rest = add_rest,
                                                crossed = crossed)

    # get the observed data averaged over region
    obs_df = average_rois(Y, info=info,
                              atlas_space = atlas_space,
                              atlas_roi = cerebellum_roi,
                              roi_selected=cerebellum_roi_selected,
                              unite_struct = False,
                              var = "Y")


    pred_df = average_rois(Yhat, info=info,
                              atlas_space = atlas_space,
                              atlas_roi = cerebellum_roi,
                              roi_selected=cerebellum_roi_selected,
                              unite_struct = False,
                              var = "X")

    # merge the dataframes, ignoring common columns (columns including task/region info)
    summary_df = pd.merge(left=obs_df, right=pred_df, how='inner')
    return summary_df

