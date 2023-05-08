"""
script to prepare dataframe for scatterplots
@ Ladan Shahshahani Joern Diedrichsen Feb 19 2023
"""
from pathlib import Path
import pandas as pd
from collections import defaultdict
import deepdish as dd
import numpy as np

import nibabel as nb
import Functional_Fusion.dataset as fdata 
import Functional_Fusion.atlas_map as am
import selective_recruitment.recruite_ana as ra
import selective_recruitment.globals as gl
import selective_recruitment.region as sroi

out_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'
if not Path(out_dir).exists():
    out_dir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'

def get_summary_data(dataset = "WMFS", 
                     ses_id = 'ses-02', 
                     atlas_space = "fs32k",
                     atlas_roi = "glasser",
                     type = "CondHalf", 
                     unite_struct = False,
                     add_rest = True):
    """
    """
    # get data
    tensor, info, _ = fdata.get_dataset(gl.base_dir,
                                        dataset,atlas= atlas_space,
                                        sess=ses_id,
                                        type=type, 
                                        info_only=False)

    # get label file
    if atlas_space == "fs32k":
        labels = []
        for hemi in ['L', 'R']:
            labels.append(gl.atlas_dir + f'/tpl-fs32k/{atlas_roi}.{hemi}.label.gii')
        var = "X" # will be used when saving the dataframe

    else:
        labels = gl.atlas_dir + f'/tpl-SUIT/{atlas_roi}_dseg.nii'
        var = "Y"

    # get average data per parcel
    parcel_data, ainfo, parcel_labels = ra.agg_data(tensor, atlas_space, labels, unite_struct = unite_struct)
    
    # use lookuptable to get region info
    region_info = sroi.get_label_names(atlas_roi, atlas_space= atlas_space) 
    
    # add rest condition for control?
    if add_rest:
        parcel_data,info = ra.add_rest_to_data(parcel_data,info)

    # Transform into a dataframe with Yhat and Y data 
    n_subj,n_cond,n_roi = parcel_data.shape

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
                     atlas_space = "SUIT3", 
                     cerebellum_roi = "Verbal2Back", 
                     cortex_roi = "Icosahedron1002",
                     type = "CondHalf", 
                     add_rest = True,
                     conn_dataset = "MDTB",
                     conn_method = "L2Regression", 
                     log_alpha = 8, 
                     crossed = True, 
                     conn_ses_id = "ses-s1"):

    """
    Function to get summary dataframe using connectivity model to predict cerebellar activation.
    It's written similar to get_symmary from recruite_ana code
    """
    
    tensor_cerebellum, info, _ = fdata.get_dataset(gl.base_dir,dataset,atlas=atlas_space,sess=ses_id,type=type, info_only=False)
    tensor_cortex, info, _ = fdata.get_dataset(gl.base_dir,dataset,atlas="fs32k",sess=ses_id,type=type, info_only=False)

    # get connectivity weights and scaling 
    conn_dir = gl.conn_dir + f"/{atlas_space}/train/{conn_dataset}_{conn_ses_id}_{cortex_roi}_{conn_method}"

    # load the model averaged over subjects
    fname = conn_dir + f"/{conn_dataset}_{conn_ses_id}_{cortex_roi}_{conn_method}_A{log_alpha}_avg.h5"
    model = dd.io.load(fname) 
    # weights = model.coef_
    # scale = model.scale_ 

    # prepare the cortical data 
    # NOTE: to use connectivity weights estimated in MDTB you always need to pass a tesselation
    # BECAUSE the models have been trained using tesselations
    cortex_label = []
    for hemi in ['L', 'R']:
        cortex_label.append(gl.atlas_dir + '/tpl-fs32k' + f'/{cortex_roi}.{hemi}.label.gii')
    X_parcel, ainfo, X_parcel_labels = ra.agg_data(tensor_cortex, "fs32k", cortex_label, unite_struct = False)

    # use cortical data to predict cerebellar data (voxel-wise)
    atlas_cereb, _ = am.get_atlas("SUIT3",gl.atlas_dir)
    # Yhat = ra.predict_cerebellum(weights, scale, X_parcel, atlas_cereb, info, fwhm = 0)
    if crossed:
        X_parcel = np.concatenate([X_parcel[:, info.half == 2, :], X_parcel[:, info.half == 1, :]], axis=1)
    Yhat = model.predict(X_parcel)
    # get the cerebellar data
    # NOTE: if None is passed, then it will average over the whole cerebellum
    if cerebellum_roi is not None:
        cerebellum_label = gl.atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum_roi}_space-SUIT_dseg.nii'
        # use lookuptable to get region info
        region_info = sroi.get_label_names(cerebellum_roi) 
        # get observed cerebellar data
        Y_parcel, ainfo, Y_parcel_labels = ra.agg_data(tensor_cerebellum, "SUIT3", cerebellum_label, unite_struct = False)

        # get predicted cerebellar data
        Yhat_parcel, ainfo, Yhat_parcel_labels = ra.agg_data(Yhat, "SUIT3", cerebellum_label, unite_struct = False)

    else: 
        # there's only one parcel: the whole cerebellum
        parcels = [1]
        # aggregate observed values over the whole cerebellum
        # aggregate predicted values over the whole cerebellum
        pass

    # add rest condition for control?
    if add_rest:
        Yhat_parcel,_ = ra.add_rest_to_data(Yhat_parcel,info)
        Y_parcel,info = ra.add_rest_to_data(Y_parcel,info)
        
    # Transform into a dataframe with Yhat and Y data 
    n_subj,n_cond,n_roi = Yhat_parcel.shape

    summary_list = [] 
    for i in range(n_subj):
        for r in range(n_roi):
            info_sub = info.copy()
            vec = np.ones((len(info_sub),))
            info_sub["sn"]    = i * vec
            info_sub["roi"]   = Yhat_parcel_labels[r] * vec
            info_sub["roi_name"] = region_info[r+1]
            info_sub["X"]     = Yhat_parcel[i,:,r]
            info_sub["Y"]     = Y_parcel[i,:,r]

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0,ignore_index=True)
    return summary_df


if __name__ == "__main__":
    D = get_summary_conn(dataset="WMFS",
                        ses_id='ses-02',
                        type="CondHalf",
                        cerebellum_roi='NettekovenSym68c32integLR',
                        cortex_roi="Icosahedron1002",
                        add_rest=True)
    pass


    




