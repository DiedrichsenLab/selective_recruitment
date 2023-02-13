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
import cortico_cereb_connectivity.data as cdata 
import selective_recruitment.recruite_ana as ra


# set base directory of the functional fusion 
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'


# run this function to make sure that you have saved data tensors
def prep_tensor(dataset_name = "WMFS", ses_id = 'ses-02'):
   cdata.save_data_tensor(dataset = dataset_name,
                         atlas='SUIT3',
                         ses_id=ses_id,
                         type="CondHalf")
   cdata.save_data_tensor(dataset = dataset_name,
                         atlas='fs32k',
                         ses_id=ses_id,
                         type="CondHalf")
   return

# prepare data for cortex and cerebellum
def get_XY(label_cereb, 
           label_cortex,
           file_path = cdata.conn_dir, 
           dataset = "WMFS", 
           ses_id = "ses-02",
           type = "CondHalf", 
           atlas_cereb = "SUIT3",
           atlas_cortex = "fs32k",
           unite_struct = False
           ):
    # get dataset class object
    Data = fdata.get_dataset_class(base_dir, dataset=dataset)

    # get info
    info = Data.get_info(ses_id,type)

    # load data tensor for SUIT3 (TODO: add an option to specify atlases other than SUIT3 and fs32k)
    file_suit = file_path + f'/{dataset}/{dataset}_SUIT3_{ses_id}_{type}.npy'
    Ydat = np.load(file_suit)

    # load data tensor for fs32k
    file_fs32k = file_path + f'/{dataset}/{dataset}_fs32k_{ses_id}_{type}.npy'
    Xdat = np.load(file_fs32k)
    
    # create instances of atlases for the cerebellum and cortex
    atlas_cereb, ainfo = am.get_atlas('SUIT3',atlas_dir)
    atlas_cortex, ainfo = am.get_atlas('fs32k', atlas_dir)

    # get parcel for both atlases
    atlas_cereb.get_parcel(label_cereb)
    atlas_cortex.get_parcel(label_cortex, unite_struct = unite_struct)

    # aggregate data over parcels
    X_parcel = fdata.agg_parcels(Xdat , atlas_cortex.label_vector,fcn=np.nanmean)
    Y_parcel = fdata.agg_parcels(Ydat , atlas_cereb.label_vector,fcn=np.nanmean)
    return X_parcel, Y_parcel, info

def get_summary_whole(outpath = cdata.conn_dir, 
                      dataset = "WMFS", 
                      ses_id = 'ses-02', 
                      type = "CondHalf"):

    """
    get summary dataframe to plot scatterplot for average ceerbellum vs average cortex
    Args: 
        outpath (str)      - path to save the file
        dataset_name (str) - name of the dataset (as is used in functional fusion framework)
        ses_id (str)       - name of the session in the dataset you want to use

    Returns:
        summ_df (pd.DataFrame) - summary dataframe to be saved and used for plotting
    """

    # get label files for cerebellum and cortex
    # NOTE: To average over cerebellum or cortex, pass on masks as label files
    label_cereb = atlas_dir + '/tpl-SUIT/' + 'tpl-SUIT_res-3_gmcmask.nii'
    label_cortex = []
    for hemi in ['L', 'R']:
        label_cortex.append(atlas_dir + '/tpl-fs32k' + f'/tpl-fs32k_hemi-{hemi}_mask.label.gii')

    X_parcel, Y_parcel, info = get_XY(label_cereb, 
                                      label_cortex,
                                      file_path = cdata.conn_dir, 
                                      dataset = dataset, 
                                      ses_id = ses_id,
                                      atlas_cereb = "SUIT3",
                                      atlas_cortex = "fs32k",
                                      type = type, 
                                      unite_struct=True
                                     )

    summ_df = ra.run_regress(X_parcel,Y_parcel,info,fit_intercept = False)

    # save the dataframe
    filepath = os.path.join(outpath, dataset, f'sc_{dataset}_{ses_id}_whole.tsv')
    summ_df.to_csv(filepath, index = False, sep='\t')

    return summ_df

def get_summary_roi(outpath = cdata.conn_dir, 
                    dataset = "WMFS", 
                    ses_id = 'ses-02', 
                    cerebellum = "Verbal2Back", 
                    cortex = "Verbal2Back.32k", 
                    type = "CondHalf"):

    """
    get summary dataframe for roi-wise analysis
    """

    # get label files for cerebellum and cortex
    # NOTE: To average over cerebellum or cortex, pass on masks as label files
    label_cereb = atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum}_space-SUIT_dseg.nii'
    label_cortex = []
    for hemi in ['L', 'R']:
        label_cortex.append(atlas_dir + '/tpl-fs32k' + f'/{cortex}.{hemi}.label.gii')

    X_parcel, Y_parcel, info = get_XY(label_cereb, 
                                      label_cortex,
                                      file_path = cdata.conn_dir, 
                                      dataset = dataset, 
                                      ses_id = ses_id,
                                      atlas_cereb = "SUIT3",
                                      atlas_cortex = "fs32k",
                                      type = type, 
                                      unite_struct=True
                                     )

    summ_df = ra.run_regress(X_parcel,Y_parcel,info,fit_intercept = False)

    # save the dataframe
    filepath = os.path.join(outpath, dataset, f'sc_{dataset}_{ses_id}_{cerebellum}.tsv')
    summ_df.to_csv(filepath, index = False, sep='\t')
    return

def get_summary_conn(outpath = cdata.conn_dir, 
                     dataset = "WMFS", 
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
    X_parcel, Y_parcel, info = get_XY(label_cereb, 
                                      label_cortex,
                                      file_path = cdata.conn_dir, 
                                      dataset = dataset, 
                                      ses_id = ses_id,
                                      atlas_cereb = "SUIT3",
                                      atlas_cortex = "fs32k",
                                      type = type,
                                      unite_struct=False
                                     )
    # use connectivity weights to predict
    Yhat = ra.predict_cerebellum(weights, scale, X_parcel, atlas_cereb, info, fwhm = 0)

    # aggregate values over voxel within parcel
    Yhat_parcel = fdata.agg_parcels(Yhat , atlas_cereb.label_vector,fcn=np.nanmean)

    # run regression (use Yhat as X)
    summ_df_list = []
    # loop over parcels
    for p in range(Yhat_parcel.shape[2]):
        summ_df_list.append(ra.run_regress(Yhat_parcel[:, :, p],Y_parcel[:, :, p],info,fit_intercept = False))

    # concatenate dataframe
    summ_df = pd.concat(summ_df_list)

    # save the dataframe
    filepath = os.path.join(outpath, dataset, f'sc_{dataset}_{ses_id}_conn.tsv')
    summ_df.to_csv(filepath, index = False, sep='\t')


    return summ_df


if __name__ == "__main__":
    # prep_tensor(dataset_name = "WMFS", ses_id = "ses-01")
    # prep_tensor(dataset_name = "WMFS", ses_id = "ses-01")

    get_summary_whole(outpath = cdata.conn_dir, 
                      dataset = "WMFS", 
                      ses_id = 'ses-02', 
                      type = "CondHalf")
    
