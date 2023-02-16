"""
script to prepare dataframe for scatterplots
@ Ladan Shahshahani Jan 30 2023 12:57
"""
import os
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict

import nibabel as nb
import Functional_Fusion.dataset as fdata 
import Functional_Fusion.atlas_map as am
# import cortico_cereb_connectivity.data as cdata 
import selective_recruitment.recruite_ana as ra


# set base directory of the functional fusion 
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'

outdir = '/Volumes/diedrichsen_data$/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'

def agg_data(tensor, atlas, label, unite_struct = True):
    """
    aggregates the data ready for regression over parcels or entire structure
    """
    # get data tensor
    atlas, ainfo = am.get_atlas(atlas,atlas_dir)

    # NOTE: atlas.get_parcel takes in path to the label file, not an array
    if (isinstance(atlas, am.AtlasSurface)) | (isinstance(atlas, am.AtlasSurfaceSymmetric)):
        if label is not None:
            atlas.get_parcel(label, unite_struct = unite_struct)
        else: # passes on mask to get parcel if you want the average across the whole structure
            atlas.get_parcel(ainfo['mask'], unite_struct) 

    else:
        if label is not None:
            atlas.get_parcel(label)
        else: # passes on mask to get parcel if you want the average across the whole structure
            atlas.get_parcel(ainfo['mask']) 


    # aggregate over voxels/vertices within parcels
    data, parcel_labels = fdata.agg_parcels(tensor , atlas.label_vector, fcn=np.nanmean)
    return data, ainfo, parcel_labels

def add_rest(X,Y,info):
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
    a = info.iloc[:1]
    a['cond_name']='rest'
    a['reg_id']=max(info.reg_id)+1
    info_new = pd.concat([info,pd.DataFrame(a)],ignore_index=True)
    return X_new,Y_new,info_new

def get_summary(dataset = "WMFS", 
                ses_id = 'ses-02', 
                type = "CondAll", 
                cerebellum_space = 'SUIT3',
                cortex_space = 'fs32k',
                cerebellum_roi =None,
                cortex_roi = None):
    """
    get summary dataframe to plot scatterplot for cerebellum vs. cortex
    Args: 
        dataset_name (str) - name of the dataset (as is used in functional fusion framework)
        ses_id (str)       - name of the session in the dataset you want to use

    Returns:
        sum_df (pd.DataFrame) - summary dataframe to be saved and used for plotting
    """
    # Get the dataset from Functional Fusion framework
    tensor_cerebellum, info, _ = fdata.get_dataset(base_dir,
            dataset,
            atlas=cerebellum_space,
            sess=ses_id,
            type=type)
    tensor_cortex, info, _ = fdata.get_dataset(base_dir,
            dataset,
            atlas=cortex_space,
            sess=ses_id,
            type=type)
    
    # get label files for cerebellum and cortex
    ## if None is passed then it will be averaged over the whole
    if cerebellum_roi is not None:
        cerebellum_roi = atlas_dir + '/' + cerebellum_roi + '_dseg.nii'
    if cortex_roi is not None:
        cortex_label = []
        # Ultimately replace this with label CIFTI, to avoid additional code-writing
        for hemi in ['L', 'R']:
            cortex_label.append(atlas_dir + '/' + cortex_roi + f'.{hemi}.label.gii')
        cortex_roi = cortex_label

    # get the data for all the subjects for cerebellum
    Y_parcel, ainfo, cerebellum_parcel = agg_data(tensor_cerebellum, 
            atlas = cerebellum_space, 
            label = cerebellum_roi)

    X_parcel, ainfo, cortex_parcel = agg_data(tensor_cortex, 
            atlas = cortex_space, 
            label = cortex_roi)
    
    # do regression and get the summary DataFrame
    X_parcel,Y_parcel,info_new = add_rest(X_parcel,Y_parcel,info)
    D = ra.run_regress(X_parcel,Y_parcel,info_new,
            fit_intercept = True)
    return D

def get_summary_conn(dataset = "WMFS", 
                     ses_id = 'ses-02', 
                     cerebellum = "Verbal2Back", 
                     cortex = "Icosahedron-1002_Sym.32k",
                     type = "CondHalf", 
                     conn_dataset = "MDTB", 
                     conn_method = "L2Regression", 
                     log_alpha = 8, 
                     conn_ses_id = "ses-s1"):

    """
    """
    
    tensor_cerebellum, info, _ = fdata.get_dataset(base_dir,dataset,atlas="SUIT3",sess=ses_id,type=type, info_only=False)
    tensor_cortex, info, _ = fdata.get_dataset(base_dir,dataset,atlas="fs32k",sess=ses_id,type=type, info_only=False)
    
    # get label files for cerebellum and cortex
    # NOTE: To average over cerebellum or cortex, pass on masks as label files
    label_cereb = atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum}_space-SUIT_dseg.nii'
    label_cortex = []
    for hemi in ['L', 'R']:
        label_cortex.append(atlas_dir + '/tpl-fs32k' + f'/{cortex}.{hemi}.label.gii')

    # get connectivity weights and scaling 
    conn_dir = os.path.join(cdata.conn_dir, conn_dataset, "train")
    weights = np.load(os.path.join(conn_dir, f"{cortex}_{conn_ses_id}_{conn_method}_logalpha_{log_alpha}_best_weights.npy"))
    scale = np.load(os.path.join(conn_dir, f'{conn_dataset}_scale.npy'))

    atlas_cereb, ainfo = am.get_atlas('SUIT3',atlas_dir)
    atlas_cereb.get_parcel(label_cereb)
    # get label files for cerebellum and cortex
    ## if None is passed then it will be averaged over the whole
    if cerebellum is not None:
        cerebellum_label = atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum}_space-SUIT_dseg.nii'
    if cortex is not None:
        cortex_label = []
        for hemi in ['L', 'R']:
            cortex_label.append(atlas_dir + '/tpl-fs32k' + f'/{cortex}.{hemi}.label.gii')

    # get the data for all the subjects for cerebellum
    Y_parcel, info, _ = get_data(dataset = dataset, ses_id=ses_id, type = type, atlas = "SUIT3", label = cerebellum_label)

    X_parcel, info, _ = get_data(dataset = dataset, ses_id=ses_id, type = type, atlas = "fs32k", label = cortex_label)
    
    # use connectivity weights to predict
    Yhat = ra.predict_cerebellum(weights, scale, X_parcel, atlas_cereb, info, fwhm = 0)

    # aggregate values over voxel within parcel
    Yhat_parcel, parcels = get_data(Yhat, atlas = "SUIT3", label = cerebellum_label)

    # run regression (use Yhat as X)
    summ_df_list = []
    # loop over parcels
    for p in range(Yhat_parcel.shape[2]):
        D = ra.run_regress(Yhat_parcel[:, :, parcels[p]],Y_parcel[:, :, parcels[p]],info,fit_intercept = False)
        D['region'] = parcels[p] *np.ones([len(D.index), ])
        summ_df_list.append(D)

    # concatenate dataframe
    summ_df = pd.concat(summ_df_list)

    # save the dataframe
    filepath = os.path.join(outpath, dataset, f'sc_{dataset}_{ses_id}_{cerebellum}_conn.tsv')
    summ_df.to_csv(filepath, index = False, sep='\t')
    return summ_df


if __name__ == "__main__":
    
    D = get_summary(dataset = "WMFS", 
                ses_id = 'ses-02', 
                type = "CondAll", 
                cerebellum_roi ='tpl-SUIT/atl-Verbal2Back_space-SUIT', 
                cortex_roi = "tpl-fs32k/Verbal2Back.32k")
    pass
