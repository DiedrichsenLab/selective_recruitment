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

import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am

import cortico_cereb_connectivity.globals as ccc_gl

import selective_recruitment.globals as gl
import selective_recruitment.regress as ra
import selective_recruitment.region as sroi

def add_rest_to_data(X,info):
    """ adds rest to X and Y data matrices and info
    Args:
        X (ndarray): n_subj,n_cond,n_reg data matrix  
        info (DataFrame): Dataframe with information 

    Returns:
        X (ndarray): n_subj,n_cond+1,n_reg data matrix  
        info (DataFrame): Dataframe with information 
    """
    n_subj,n_cond,n_reg = X.shape
    X_new = np.zeros((n_subj,n_cond+1,n_reg))
    X_new[:,:-1,:]=X
    a = pd.DataFrame({'cond_name':'rest',
                'reg_id':max(info.reg_id)+1,
                'cond_num':max(info.reg_id)+1},index=[0])
    info_new = pd.concat([info,pd.DataFrame(a)],ignore_index=True)
    return X_new,info_new

def get_voxdata_obs_pred(dataset = "WMFS", 
                         ses_id = 'ses-02',
                         subj = None,
                         atlas_space='SUIT3',
                         cortex = 'Icosahedron1002',
                         type = "CondAll",
                         mname_base = "MDTB_ses-s1",
                         mmethod = "L2Regression_A8",
                         add_rest = False, 
                         crossed = True):
    """gets the obsesrved and predicted voxel data for a given dataset and model.

    Args:
        dataset (str, optional): name of the dataset to make predictions for. Defaults to "WMFS".
        ses_id (str, optional): string representing the session id. Defaults to 'ses-02'.
        subj (list or None, optional): list of names of the subjects. Defaults to None: does for all subjects.
        atlas_space (str, optional): atlas space you want to have the data in. Defaults to 'SUIT3'.
        cortex (str, optional): name of the label to be used for the cortex. Defaults to 'Icosahedron1002'.
        type (str, optional): type of the data you want to use. Defaults to "CondAll": for averaged across all runs.
        mname_base (str, optional): model name base containing the modelling data set and session id. Defaults to "MDTB_ses-s1".
        mmethod (str, optional): modeling method. Defaults to "L2Regression".
        alpha (str, optional): alpha used in modeling as a string. Defaults to "A8".
        add_rest (bool, optional): add fake rest (all zeros)? Defaults to False.
        crossed (bool, optional): flip the order of halfs (type has to be CondHalf)? Defaults to True.
        
    Returns:
        Y (np.ndarray): observed cerebellar data (n_subj,n_cond,n_vox)
        YP (np.ndarray): predicted cerebellar data (n_subj,n_cond,n_vox)
        atlas (Object): cortical atlas
        info (pd.DataFrame): dataframe with info for data 
    """
    
    # get observed cerebellar data
    Y,info,dset = ds.get_dataset(gl.base_dir,dataset,
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)
    
    # get the connectivity model
    ## first get alpha and method name
    method, alpha = mmethod.split("_")
    mname = f"{mname_base}_{cortex}_{method}"
    model_path = os.path.join(ccc_gl.conn_dir,atlas_space,'train',mname)
    fname = model_path + f"/{mname}_{alpha}_avg.h5" # get the model averaged over subjects
    json_name = model_path + f"/{mname}_{alpha}_avg.json"
    conn_model = dd.io.load(fname)
    
    # get cortical data to be used as input to the connectivity model
    X,info,dset = ds.get_dataset(gl.base_dir,dataset,
                                    atlas="fs32k", # for cortex we always use fs32k
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)
    atlas,ainf = am.get_atlas('fs32k',gl.atlas_dir)
    # make strings for the cortical label files
    ## label files are saved as gifti, separated by hemisphere
    label=[gl.atlas_dir+'/tpl-fs32k/'+cortex+'.L.label.gii',
           gl.atlas_dir+'/tpl-fs32k/'+cortex+'.R.label.gii']
    atlas.get_parcel(label,unite_struct=False)
    
    X, _ = ds.agg_parcels(X, atlas.label_vector, fcn=np.nanmean)
    
    if crossed: # flip the order of the halfs
        if type == "CondHalf":
            X_parcel = np.concatenate([X[:, info.half == 2, :], X[:, info.half == 1, :]], axis=1)
        else:
            # leaves X_parcel unchanged
            pass

    # use cortical data and the model to get the cerebellar predicted data
    YP = conn_model.predict(X)
    if add_rest:
        Y,_ = ra.add_rest_to_data(Y,info)
        YP,info = ra.add_rest_to_data(YP,info)
    
    return Y,YP,atlas,info

def get_summary_data(tensor, 
                     atlas_space = "fs32k",
                     atlas_roi = "glasser",
                     unite_struct = False,
                     add_rest = True, 
                     var = "Y"):
    """makes a summary dataframe for data averaged over voxels in a parcel

    Args:
        tensor (np.ndarray): n_subj,n_cond,n_vox data matrix
        atlas_space (str, optional): _description_. Defaults to "fs32k".
        atlas_roi (str, optional): _description_. Defaults to "glasser".
        type (str, optional): _description_. Defaults to "CondHalf".
        unite_struct (bool, optional): _description_. Defaults to False.
        add_rest (bool, optional): _description_. Defaults to True.
        var (str, optional): column name to be used in the summary dataframe. Defaults to "Y".

    Returns:
        summary_df (pd.Dataframe): a dataframe with the summary data for each parcecl within atlas_roi
    """

    # get label file
    if atlas_space == "fs32k":
        atlas_dir = f'{gl.atlas_dir}/tpl-fs32k'
        labels = []
        for hemi in ['L', 'R']:
            labels.append(atlas_dir + f'/{atlas_roi}.{hemi}.label.gii')
    else:
        atlas_dir = f'{gl.atlas_dir}/tpl-SUIT'
        labels = f'{atlas_dir}/{atlas_roi}_dseg.nii'
        
    # use lookuptable to get the names of the regions if lut file exists
    if os.exists(f"{atlas_dir}/{atlas_roi}.lut"):
        region_info = sroi.get_label_names(atlas_roi, atlas_space= atlas_space)
    else: # if lut file doesn't exist, just use the parcel labels
        region_info = [f"parcel_{i}" for i in parcel_labels]
        
    # get average data per parcel
    # get atlas
    atlas, ainfo = am.get_atlas(atlas,gl.atlas_dir)

    # NOTE: atlas.get_parcel takes in path to the label file, not an array
    if (isinstance(atlas, am.AtlasSurface)) | (isinstance(atlas, am.AtlasSurfaceSymmetric)):
        if labels is not None:
            atlas.get_parcel(labels, unite_struct = unite_struct)
            
        else: # passes on mask to get parcel if you want the average across the whole structure
            atlas.label_vector = np.ones((atlas.P,),dtype=int)

    else:
        if labels is not None:
            atlas.get_parcel(labels)
        else: # passes on mask to get parcel if you want the average across the whole structure
            atlas.label_vector = np.ones((atlas.P,),dtype=int)
    # aggregate over voxels/vertices within parcels
    parcel_data, parcel_labels = ds.agg_parcels(tensor , 
                                                atlas.label_vector, 
                                                fcn=np.nanmean)

    # add rest condition for control?
    if add_rest:
        parcel_data,info = ra.add_rest_to_data(parcel_data,info)

    # Transform into a dataframe
    n_subj, n_cond, n_roi = parcel_data.shape

    summary_list = [] 
    for i in range(n_subj):
        for r in range(n_roi):
            info_sub = info.copy()
            vec = np.ones((len(info_sub),))
            info_sub["sn"]    = i * vec
            info_sub["roi"]   = parcel_labels[r] * vec
            info_sub["roi_name"] = region_info[r+1]
            info_sub[var]     = parcel_data[i,:,r]

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0,ignore_index=True)

    return summary_df

def get_summary_conn(dataset = "WMFS", 
                     ses_id = 'ses-02',
                     subj = None, 
                     atlas_space = "SUIT3", 
                     cerebellum_roi = "Verbal2Back", 
                     cortex_roi = "Icosahedron1002",
                     type = "CondHalf", 
                     add_rest = True,
                     mname_base = "MDTB_ses-s1",
                     mmethod = "L2Regression_A8", 
                     crossed = True):

    """
    Function to get summary dataframe using connectivity model to predict cerebellar activation.
    It's written similar to get_symmary from recruite_ana code
    """
    
    Y,Yhat, _, _ = get_voxdata_obs_pred(dataset = dataset, 
                                            ses_id = ses_id,
                                            subj = subj,
                                            atlas_space=atlas_space,
                                            cortex = cortex_roi,
                                            type = type,
                                            mname_base = mname_base,
                                            mmethod = mmethod,
                                            add_rest = add_rest, 
                                            crossed = crossed)
    
    # get the observed data averaged over region
    obs_df = get_summary_data(Y, 
                              atlas_space = atlas_space,
                              atlas_roi = cerebellum_roi,
                              unite_struct = False,
                              add_rest = add_rest, 
                              var = "Y")
    
    pred_df = get_summary_data(Yhat, 
                              atlas_space = atlas_space,
                              atlas_roi = cerebellum_roi,
                              unite_struct = False,
                              add_rest = add_rest, 
                              var = "X")
    # merge the dataframes, ignoring common columns (columns including task/region info)
    summary_df = pd.merge(left=obs_df, right=pred_df, how='inner')
    return summary_df


if __name__ == "__main__":
    # test cases for debugging
    y, yp, atlas, info = get_voxdata_obs_pred(dataset = "WMFS", 
                                                ses_id = 'ses-02',
                                                subj = None,
                                                atlas_space='SUIT3',
                                                cortex = 'Icosahedron1002',
                                                type = "CondAll",
                                                mname_base = "MDTB_ses-s1",
                                                mmethod = "L2Regression_A8",
                                                add_rest = False, 
                                                crossed = True)
    pass

