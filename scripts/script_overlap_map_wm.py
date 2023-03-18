#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to make label files for the cortical and cerebellar rois

Created on 07/03/2023
Author: Ladan Shahshahani
"""
# import packages
import numpy as np
import pandas as pd
from pathlib import Path
# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import selective_recruitment.globals as gl
import selective_recruitment.recruite_ana as ra
import cortico_cereb_connectivity.evaluation as ccev
import SUITPy as suit
import surfAnalysisPy as sa
import matplotlib.pyplot as plt 
#
import os
import nibabel as nb
import nitools as nt
import SUITPy.flatmap as flatmap
import scipy.stats as ss

# TODO: use tesselations?
def get_contrast(data, info):
    """calculates the contrasts for load and recall effect

    Args:
        data (np.ndarray): 
        info (pd.DataFrame): _pandas dataframe representing conditions_ 
        load_effect (int, optional): _value of the load to be used as effect_. Defaults to 6.
        load_baseline (int, optional): _value of the load to be used as baseline_. Defaults to 2.
        recall_effect (int, optional): _value of the recall to be used as effect_. Defaults to 0.
        recall_baseline (int, optional): _value of the recall to be used as baseline_. Defaults to 1.
        
    Returns:
        data_load
        data_recall
    """
    ## get the index for baseline condition
    idx_base_load = info.load == 2
    ## get the index for effect condition
    idx_effect_load = info.load == 6
    ## load effect
    data_load = np.nanmean(data[idx_effect_load, :], axis = 0) - np.nanmean(data[idx_base_load, :], axis = 0)
    
    # calculate the effect of recall direction
    ## get the index for baseline condition
    idx_base_recall = info.recall == 1
    ## get the index for effect condition
    idx_effect_recall = info.recall == 0
    ## recall effect
    data_recall = np.nanmean(data[idx_effect_recall, :], axis = 0) - np.nanmean(data[idx_base_recall, :], axis = 0)
    return data_load, data_recall


def load_contrast(ses_id = 'ses-02',
                subj = "group",atlas_space='SUIT3',
                phase=0):
    """
    1. Gets group data 
    2. Calculates the desired contrast (uses the info file to get conditions)
    """    
    data,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj)
    data = data[0]

    # loop over phases
    if phase == "overall":
        idx = True*np.ones([len(info,)], dtype=bool)
    else:
        idx = (info.phase == phase)
        
    info = info.loc[idx]
    data = data[idx, :]
    
    # get contrasts
    data_load, data_recall = get_contrast(data, info)
    return data_load,data_recall

def plot_overlap_cerebellum():
    """
    Makes an overlap plot for the cerebellum 
    """
    load_eff,dir_eff=load_contrast(ses_id = 'ses-02',subj = "group",atlas_space='SUIT3',phase=1)
    data=np.c_[dir_eff,
               np.zeros(load_eff.shape),
               load_eff].T # Leave the green gun empty 
    atlas, a_info = am.get_atlas('SUIT3',gl.atlas_dir)
    Nii = atlas.data_to_nifti(data)
    data = suit.vol_to_surf(Nii,space='SUIT')
    rgb = suit.flatmap.map_to_rgb(data,scale=[0.05,0.1,0.1],threshold=[0.02,0.05,0.05])
    suit.flatmap.plot(rgb,overlay_type='rgb')
    pass


def plot_overlap_cortex(phase=0):
    """
    Makes an overlap plot for the cerebellum 
    """
    load_eff,dir_eff=load_contrast(ses_id = 'ses-02',subj = "group",atlas_space='fs32k',phase=phase)
    data=np.c_[dir_eff,
               np.zeros(load_eff.shape),
               load_eff] # Leave the green gun empty 
    rgb = suit.flatmap.map_to_rgb(data,scale=[0.05,0.1,0.1],threshold=[0.02,0.05,0.05])

    atlas, a_info = am.get_atlas('fs32k',gl.atlas_dir)
    for i,h in enumerate(['L']):
        # plt.subplot(1,2,i+1)
        rgb_surf=np.zeros((atlas.vertex[i].max()+1,4))*np.nan
        rgb_surf[atlas.vertex[i],:]=rgb[atlas.indx_full[i],:]
        sa.plot.plotmap(rgb,'fs32k_' + h,overlay_type='rgb')
    pass


def calculate_R(A, B):
    """Calculates correlation between A and B without subtracting the mean.

    Args:
        A (nd-array):
        B (nd-array):
    Returns:
        R (scalar): Correlation between A and B
    """
    SYP = np.nansum(A * B, axis=0)
    SPP = np.nansum(B * B, axis=0)
    SST = np.nansum(A ** 2, axis=0)  # use np.nanmean(Y) here?

    R = np.nansum(SYP) / (np.sqrt(np.nansum(SST) * np.nansum(SPP)))
    return R

def overlap_corr(dataset = 'WMFS', 
                            type = 'CondAll',
                            ses_id = 'ses-02',
                            subject = "group",
                            atlas_space = 'SUIT3', 
                            threshold = 80, 
                            binarize = True,
                            ):
    """
    quantifying the overlap between the two contrasts by calculating the overlap between maps
    # TODO: on group, on each subject separately
    """
    # create a dataset class for the dataset
    # get Dataset class for your dataset
    Data = ds.get_dataset_class(gl.base_dir, dataset=dataset)
    tensor = []
    tensor_cerebellum, info_tsv, _ = ds.get_dataset(gl.base_dir,dataset,atlas="SUIT3",sess=ses_id,type=type, info_only=False)
    tensor_cortex, info_tsv, _ = ds.get_dataset(gl.base_dir,dataset,atlas="fs32k",sess=ses_id,type=type, info_only=False)
    
    tensor.append(tensor_cerebellum)
    tensor.append(tensor_cortex)
    
    # smoothing with kernel 3
    atlas, a_info = am.get_atlas(atlas_space,gl.atlas_dir)
    # smat = ra.get_smooth_matrix(atlas, fwhm =3)

    # # smooth data
    # cifti_data = cifti_data@smat
    T = Data.get_participants()
    
    # loop over phases
    DD = []
    for i, sub in enumerate(T.participant_id):
        for p, phase in enumerate(["Enc", "Ret", "overall"]):
            # get the index for the phase
            if phase == "overall":
                idx = True*np.ones([len(info_tsv,)], dtype=bool)
            else:
                idx = info_tsv.phase == p
                
            info = info_tsv.loc[idx]

            for ss, structure in enumerate(["cerebellum", "cortex"]):
                data = tensor[ss][i, idx, :]
                # get contrasts
                data_load, data_recall = get_contrast(data, info)
                
                # threshold the data
                thresh_load =ra.threshold_map(data_load, threshold, binarize = binarize)
                thresh_recall =ra.threshold_map(data_recall, threshold, binarize = binarize)

                # calculate correlations 
                R = calculate_R(thresh_load, thresh_recall)
                R_dict = {}
                R_dict["dataset"] = dataset
                R_dict["ses_id"] =ses_id
                R_dict["phase"] = phase
                R_dict["structure"] = structure
                R_dict["R"] = R
                R_dict["sn"] = sub
                R_df = pd.DataFrame(R_dict, index = [0])

                DD.append(R_df)
    return pd.concat(DD)

if __name__=="__main__":
    plot_overlap_cortex()