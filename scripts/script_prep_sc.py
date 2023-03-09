"""
script to prepare dataframe for scatterplots
@ Ladan Shahshahani Joern Diedrichsen Feb 19 2023
"""
import os
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict

import nibabel as nb
import Functional_Fusion.dataset as fdata 
import Functional_Fusion.atlas_map as am
import selective_recruitment.recruite_ana as ra
import selective_recruitment.globals as gl

out_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'
if not Path(out_dir).exists():
    out_dir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'


def get_summary_conn(dataset = "WMFS", 
                     ses_id = 'ses-02', 
                     cerebellum_roi = "Verbal2Back", 
                     cortex_roi = "Icosahedron-1002_Sym.32k",
                     type = "CondHalf", 
                     add_rest = True,
                     conn_dataset = "MDTB", 
                     conn_method = "L2Regression", 
                     log_alpha = 8, 
                     conn_ses_id = "ses-s1"):

    """
    Function to get summary dataframe using connectivity model to predict cerebellar activation.
    It's written similar to get_symmary from recruite_ana code
    """
    
    tensor_cerebellum, info, _ = fdata.get_dataset(gl.base_dir,dataset,atlas="SUIT3",sess=ses_id,type=type, info_only=False)
    tensor_cortex, info, _ = fdata.get_dataset(gl.base_dir,dataset,atlas="fs32k",sess=ses_id,type=type, info_only=False)

    # get connectivity weights and scaling 
    conn_dir = os.path.join(gl.conn_dir, conn_dataset, "train")
    weights = np.load(os.path.join(conn_dir, f"{cortex_roi}_{conn_ses_id}_{conn_method}_logalpha_{log_alpha}_best_weights.npy"))
    scale = np.load(os.path.join(conn_dir, f'{conn_dataset}_scale.npy'))

    # prepare the cortical data 
    # NOTE: to use connectivity weights estimated in MDTB you always need to pass a tesselation
    # BECAUSE the models have been trained using tesselations
    cortex_label = []
    for hemi in ['L', 'R']:
        cortex_label.append(gl.atlas_dir + '/tpl-fs32k' + f'/{cortex_roi}.{hemi}.label.gii')
    X_parcel, ainfo, X_parcel_labels = ra.agg_data(tensor_cortex, "fs32k", cortex_label, unite_struct = False)

    # use cortical data to predict cerebellar data (voxel-wise)
    atlas_cereb, _ = am.get_atlas("SUIT3",gl.atlas_dir)
    Yhat = ra.predict_cerebellum(weights, scale, X_parcel, atlas_cereb, info, fwhm = 0)

    # get the cerebellar data
    # NOTE: if None is passed, then it will average over the whole cerebellum
    if cerebellum_roi is not None:
        cerebellum_label = gl.atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum_roi}_space-SUIT_dseg.nii'
        

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
        Yhat_parcel,Y_parcel,info = ra.add_rest_to_data(Yhat_parcel,Y_parcel,info)
        
    # Transform into a dataframe with Yhat and Y data 
    n_subj,n_cond,n_roi = Yhat_parcel.shape

    summary_list = [] 
    for i in range(n_subj):
        for r in range(n_roi):
            info_sub = info.copy()
            vec = np.ones((len(info_sub),))
            info_sub["sn"]    = i * vec
            info_sub["roi"]   = Yhat_parcel_labels[r] * vec
            info_sub["X"]     = Yhat_parcel[i,:,r]
            info_sub["Y"]     = Y_parcel[i,:,r]

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0,ignore_index=True)
    return summary_df


if __name__ == "__main__":
    #####################################################
    # print("connectivity for hierarchical parcellation: Regression")
    # D = get_summary_conn(dataset = "WMFS", 
    #                     ses_id = 'ses-02', 
    #                     cerebellum_roi = "NettekovenSym68c32", 
    #                     cortex_roi = "Icosahedron-1002_Sym.32k",
    #                     type = "CondHalf",
    #                     add_rest=True,  
    #                     conn_dataset = "MDTB", 
    #                     conn_method = "L2Regression", 
    #                     log_alpha = 8, 
    #                     conn_ses_id = "ses-s1")

    # # do regression
    # D = ra.run_regress(D,fit_intercept=True)

    # D.to_csv(out_dir + '/ROI_NettekovenSym68c32_conn_reg.tsv',sep='\t')
    #####################################################
    print("connectivity for hierarchical parcellation: PCA")
    D = get_summary_conn(dataset = "WMFS", 
                        ses_id = 'ses-02', 
                        cerebellum_roi = "NettekovenSym68c32", 
                        cortex_roi = "Icosahedron-1002_Sym.32k",
                        type = "CondHalf",
                        add_rest=True,  
                        conn_dataset = "MDTB", 
                        conn_method = "L2Regression", 
                        log_alpha = 8, 
                        conn_ses_id = "ses-s1")

    # do regression
    D = ra.run_pca(D,zero_mean=True)

    D.to_csv(out_dir + '/ROI_NettekovenSym68c32_conn_pca.tsv',sep='\t')



    




