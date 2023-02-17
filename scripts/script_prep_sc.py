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
import selective_recruitment.globals as gl

outdir = '/Volumes/diedrichsen_data$/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'


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
    
    D = ra.get_summary(dataset = "WMFS", 
                ses_id = 'ses-02', 
                type = "CondAll", 
                cerebellum_roi ='tpl-SUIT/atl-Verbal2Back_space-SUIT', 
                cortex_roi = "tpl-fs32k/Verbal2Back.32k",
                add_rest = True)
    D = ra.run_regress(D,fit_intercept=True)
    D.to_csv(outdir + '/ROI_Verbal2back.tsv',sep='\t')
