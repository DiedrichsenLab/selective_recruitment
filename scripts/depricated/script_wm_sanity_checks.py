#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to make label files for the cortical and cerebellar rois

Created on 07/03/2023
Author: Ladan Shahshahani
"""
# import packages
import enum
from tabnanny import verbose
from tokenize import group
import numpy as np
import pandas as pd
from pathlib import Path
import os
# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as fdata
import Functional_Fusion.util as futil
import Functional_Fusion.matrix as fmatrix
import selective_recruitment.globals as gl
import regress as ra
import selective_recruitment.rsa as srsa
import cortico_cereb_connectivity.evaluation as ccev
import Correlation_estimation.util as corr_util
import PcmPy as pcm
import SUITPy as suit
# import surfAnalysisPy.stats as sas
import surfAnalysisPy as sa
import matplotlib.pyplot as plt 
from matplotlib import pyplot as plt
import seaborn as sns
from adjustText import adjust_text # to adjust the text labels in the plots (pip install adjustText)


def calc_within_subj_reliability(dataset = "WMFS", 
                                 ses_id = "ses-02", 
                                 phase = 0, 
                                 type = "CondRun",
                                 subtract_mean = False 
                                 ):

    """Calculates reliability within subjects and returns a group map
    """
    # get data tensor to calculate reliability maps
    atlas_cereb, ainfo = am.get_atlas('SUIT3', gl.atlas_dir)
    tensor, info, _ = fdata.get_dataset(
        gl.base_dir, dataset, atlas="SUIT3", sess=ses_id, type=type, info_only=False)

    X_phase = tensor[:, info.phase == phase, :]
    r_phase = fdata.reliability_within_subj(X_phase, info.run.loc[info.phase == phase], info.cond_num.loc[info.phase == phase],
                                        voxel_wise=True,
                                        subtract_mean=subtract_mean)

    # mean across runs and subjects
    r_phase_group = np.nanmean(np.nanmean(r_phase, axis=1), axis=0)
    img_nii = atlas_cereb.data_to_nifti(r_phase_group)
    img_flat = suit.flatmap.vol_to_surf([img_nii], stats='nanmean', space='SUIT')
    fig = plt.figure()
    ax = suit.flatmap.plot(data=img_flat,
                    render="matplotlib",
                    cmap="hot",
                    colorbar=True,
                    bordersize=1.5,
                    cscale=[0.2, 0.8])
    # ax.show()
    ax.set_title(f"reliability during phase {phase}")
    return r_phase, ax

def calc_between_subj_reliability(dataset = "WMFS", 
                                    ses_id = "ses-02", 
                                    phase = 0, 
                                    type = "CondRun",
                                    subtract_mean = False 
                                    ):

    # get data tensor to calculate reliability maps
    atlas_cereb, ainfo = am.get_atlas('SUIT3', gl.atlas_dir)
    tensor, info, _ = fdata.get_dataset(
        gl.base_dir, dataset, atlas="SUIT3", sess=ses_id, type=type, info_only=False)

    X_phase = tensor[:, info.phase == phase, :]
    r_phase = fdata.reliability_between_subj(X_phase, cond_vec=info.cond_num.loc[info.phase == phase],
                                            voxel_wise=True,
                                            subtract_mean=subtract_mean)
    # mean across subjects
    r_phase_group = np.nanmean(r_phase, axis=0)
    img_nii = atlas_cereb.data_to_nifti(r_phase_group)
    img_flat = suit.flatmap.vol_to_surf([img_nii], stats='nanmean', space='SUIT')
    fig = plt.figure()
    ax = suit.flatmap.plot(data=img_flat,
                    render="matplotlib",
                    cmap="hot",
                    colorbar=True,
                    bordersize=1.5,
                    cscale=[0.2, 0.8])
    # ax.show()
    ax.set_title(f"between subject reliability during phase {phase}")
    return r_phase, ax

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
    Data = fdata.get_dataset_class(gl.base_dir, dataset=dataset)

    # loop over cortex and cerebellum
    D = []
    for atlas in ["SUIT3", "fs32k"]:

        # get the data tensor
        tensor, info, _ = fdata.get_dataset(gl.base_dir,dataset,atlas = atlas, sess=ses_id,type='CondRun', info_only=False)

        # loop over conditions and calculate reliability
        for c, _ in enumerate(info.cond_name.loc[info.run == 1]):
            # get condition info data
            part_vec = info.run
            cond_vec = (info.cond_num == c+1)*(c+1)

            # get all the info from info to append to the summary dataframe
            info_cond = info.loc[(info.cond_num == c+1) & (info.run == 1)]

            r = fdata.reliability_within_subj(tensor, part_vec, cond_vec,
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
