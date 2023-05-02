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
import sys
import subprocess
# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.util as futil
import Functional_Fusion.matrix as fmatrix

import selective_recruitment.globals as gl
import selective_recruitment.recruite_ana as ra
import selective_recruitment.rsa as srsa

import Correlation_estimation.util as corr_util

import PcmPy as pcm
import SUITPy as suit
# import surfAnalysisPy.stats as sas
import surfAnalysisPy as sa
import matplotlib.pyplot as plt 
from scipy.stats import norm, ttest_1samp
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib import pyplot as plt
import seaborn as sns
from adjustText import adjust_text # to adjust the text labels in the plots (pip install adjustText)

#
import nibabel as nb
from nilearn import plotting
import nitools as nt
import nitools.cifti as ntcifti
import SUITPy.flatmap as flatmap
from nilearn import plotting
import scipy.stats as ss

import numpy as np
from numpy import matrix, sum,mean,trace,sqrt, zeros, ones
from PcmPy.matrix import indicator
from scipy.linalg import solve, pinv
from scipy.spatial import procrustes
from numpy.linalg import eigh

wkdir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'

def smooth_cifti_data(sigma = 3, 
                      atlas_space = "fs32k",
                      type = "CondAll", 
                      ses_id = "ses-02"):
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")
    # get list of subjects
    T = Data.get_participants()
    subject_list = list(T.participant_id.values)
    # subject_list.append("group")

    # get the surfaces for smoothing
    surfs = [Data.atlas_dir + f'/tpl-fs32k/tpl_fs32k_hemi-{h}_inflated.surf.gii' for i, h in enumerate(['L', 'R'])]
    # loop over subjects, load data, smooth and save with s prefix
    for s, subject in enumerate(subject_list):
        print(f"- smoothing {subject}")
        # load the cifti
        cifti_fpath = Data.data_dir.format(subject)
        cifti_fname = f"{subject}_space-{atlas_space}_{ses_id}_{type}.dscalar.nii"
        cifti_input = f"{cifti_fpath}/{cifti_fname}"

        # make up the smoothed file name
        fparts = cifti_fname.split(".")
        cifti_output = f"{cifti_fpath}/{fparts[0]}_desc-sm{int(sigma)}.{fparts[1]}.{fparts[2]}"

        # smooth the file
        ntcifti.smooth_cifti(cifti_input, 
                              cifti_output,
                              surfs[0], 
                              surfs[1], 
                              surface_sigma = sigma, 
                              volume_sigma = sigma, 
                              direction = "COLUMN", 
                              )
    return


def get_enc_ret_contrast(subj = "group", 
                        smooth = 3, 
                        atlas_space = "SUIT3",
                        type = "CondAll", 
                        recall_dir = None, 
                        ses_id = "ses-02"):
    """
    Args:
        subj (str or None) - input None for all the subjects
        smooth (bool) - True if you want to ues smoothed data
        type (str) - type of the extracted data to be used
    """
    # get dataset
    data,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type,  
                                    smooth = smooth)
    # get partitions
    partition = type[4:].lower()
    if partition == "all":
        part_vec = np.ones([len(info), ])
    else:
        part_vec = info[partition].values

    # get number of partitions
    parts = np.unique(part_vec)
    n_parts = len(parts)
    # get number of subjects 
    n_subj = data.shape[0]
    # get number of voxels
    n_vox = data.shape[2]
    # initialize arrays
    enc_data = np.zeros([n_subj, n_parts, n_vox])
    ret_data = np.zeros([n_subj, n_parts, n_vox])
    for s in range(n_subj):
        for p, pn in enumerate(parts):
            # make contrast vector for encoding and retrieval
            idx_enc = (info.phase == 0) & (part_vec == pn)
            idx_ret = (info.phase == 1) & (part_vec == pn)
            if recall_dir is not None:
                idx_enc = idx_enc & (info.recall == recall_dir)
                idx_ret = idx_ret & (info.recall == recall_dir)

            c_enc = (idx_enc*1)/np.sum(idx_enc.values*1)            
            c_ret = (idx_ret*1)/np.sum(idx_ret.values*1)
            # get contrasts
            ## enc:
            enc_data[s, p, : ] = c_enc.values.reshape(-1, 1).T@data[s]
            ## ret:
            ret_data[s, p, : ] = c_ret.values.reshape(-1, 1).T@data[s]

    return enc_data, ret_data


def get_dir_load_contrast(subj = "group", 
                        smooth = False,
                        atlas_space = "SUIT3", 
                        type = "CondAll", 
                        phase = 0, 
                        ses_id = "ses-02"):
    """
    """
    # get dataset
    data,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type,  
                                    smooth = smooth)
    # get partitions
    partition = type[4:].lower()
    if partition == "all":
        part_vec = np.ones([len(info), ])
    else:
        part_vec = info[partition].values

    # get number of partitions
    parts = np.unique(part_vec)
    n_parts = len(parts)
    # get number of subjects 
    n_subj = data.shape[0]
    # get number of voxels
    n_vox = data.shape[2]
    # initialize arrays
    dir_data = np.zeros([n_subj, n_parts, n_vox])
    load_data = np.zeros([n_subj, n_parts, n_vox])

    for s in range(n_subj):
        for p, pn in enumerate(parts):
            # make contrast vector for load and dir
            ## dir:
            idx_bw = (info.phase == phase) & (info.recall == 0) & (part_vec == pn)
            idx_fw = (info.phase == phase) & (info.recall == 1) & (part_vec == pn)
            c_dir = ((idx_bw*1)/np.sum(idx_bw.values*1))\
                     - ((idx_fw*1)/np.sum(idx_fw.values*1))
            ## load:
            # make contrast vector for load effect
            idx_load6 = (info.phase == phase) & (info.recall == 1) & (info.load == 6) & (part_vec == pn)
            idx_load2 = (info.phase == phase) & (info.recall == 1) & (info.load == 2) & (part_vec == pn)
            c_load = ((idx_load6*1)/np.sum(idx_load6.values*1))\
                     - ((idx_load2*1)/np.sum(idx_load2.values*1))
            # get contrasts
            ## load:
            load_data[s, p, : ] = c_load.values.reshape(-1, 1).T@data[s]
            ## ret:
            dir_data[s, p, : ] = c_dir.values.reshape(-1, 1).T@data[s]
    return dir_data, load_data


def get_dir_load_conj(type = "CondAll", 
                        phase = 0,
                        ses_id = "ses-02",  
                        atlas_space = "fs32k", 
                        smooth = 3):
    """
    """

    dir_data_sub, load_data_sub = get_dir_load_contrast(subj = None, 
                                                smooth = smooth,
                                                atlas_space = atlas_space, 
                                                type = type, 
                                                phase = phase, 
                                                ses_id = ses_id)
    # do test
    load_data, p_val_load = ttest_1samp(load_data_sub, axis = 0, popmean = 0, nan_policy = 'omit', alternative = 'greater')
    dir_data, p_val_dir = ttest_1samp(dir_data_sub, axis = 0, popmean = 0, nan_policy = 'omit', alternative='greater')

    return dir_data, load_data


def get_enc_ret_conj(type = "CondAll", 
                        ses_id = "ses-02",  
                        atlas_space = "fs32k", 
                        smooth = 3):

    """
    """
    enc_data_sub, ret_data_sub = get_enc_ret_contrast(subj = None, 
                                                        smooth = smooth,
                                                        atlas_space = atlas_space, 
                                                        type = type, 
                                                        ses_id = ses_id)
    # do test
    enc_data, p_val_load = ttest_1samp(enc_data_sub, axis = 0, popmean = 0, nan_policy = 'omit', alternative = 'greater')
    ret_data, p_val_dir = ttest_1samp(ret_data_sub, axis = 0, popmean = 0, nan_policy = 'omit', alternative='greater')

    return enc_data, ret_data


def get_enc_ret_rel_summ(subj = None, 
                        smooth = 3, 
                        subtract_mean = True, 
                        atlas_spaces = ["SUIT3", "fs32k"],
                        recall_dirs = [None], 
                        type = "CondHalf", 
                        ses_id = "ses-02", 
                        verbose = False):

    """
    """
    # get dataset class
    dset = ds.get_dataset_class(gl.base_dir, "WMFS")
    D = []
    for atlas in atlas_spaces:
        for dcall in recall_dirs:
            # get contrasts 
            ## data[0]: enc effect, data[1]: ret effect
            data_list = get_enc_ret_contrast(subj = subj, 
                                        smooth = smooth, 
                                        atlas_space = atlas,
                                        type = type, 
                                        recall_dir = dcall, 
                                        ses_id = ses_id)

            
            
            for p, phase in enumerate(["enc", "ret"]):
                # calculate reliabilities
                con_data = data_list[p]
                alpha_list = []
                for s in range(con_data.shape[0]):
                    dat = np.nan_to_num(con_data[s])
                    # subtract_mean?
                    if subtract_mean:
                        dat = dat - np.nanmean(dat, axis = 1, keepdims=True)

                    alpha_list.append(corr_util.calc_cronbach(dat))

                # making the summary dataframe
                ## get list of subjects
                subj_list = dset.get_participants().participant_id
                df_tmp = pd.DataFrame()
                df_tmp["sn"] = subj_list
                df_tmp["phase"] = phase
                df_tmp["effect"] = phase
                df_tmp["recall_dir"] = str(dcall)
                df_tmp["alpha"] = alpha_list
                df_tmp["atlas"] = atlas
                D.append(df_tmp)
    
    return pd.concat(D)


def get_dir_load_rel_summ(subj = None, 
                          smooth = 3, 
                          subtract_mean = True, 
                          atlas_spaces = ["SUIT3", "fs32k"],
                          phases = ["enc", "ret"], 
                          type = "CondHalf", 
                          ses_id = "ses-02", 
                          verbose = False):

    """
    """
    # get dataset class
    dset = ds.get_dataset_class(gl.base_dir, "WMFS")
    D = []
    for atlas in atlas_spaces:
        for p, phase in enumerate(phases):
            # get contrasts 
            ## data[0]: dir effect, data[1]: load effect
            data_list = get_dir_load_contrast(subj = subj, 
                                        smooth = smooth, 
                                        atlas_space = atlas,
                                        type = type, 
                                        phase = p, 
                                        ses_id = ses_id)
            
            for e, effect in enumerate(["dir", "load"]):
                # calculate reliabilities
                con_data = data_list[e]
                alpha_list = []
                for s in range(con_data.shape[0]):
                    dat = np.nan_to_num(con_data[s])
                    # subtract_mean?
                    if subtract_mean:
                        dat = dat - np.nanmean(dat, axis = 1, keepdims=True)

                    alpha_list.append(corr_util.calc_cronbach(dat))

                # making the summary dataframe
                ## get list of subjects
                subj_list = dset.get_participants().participant_id
                df_tmp = pd.DataFrame()
                df_tmp["sn"] = subj_list
                df_tmp["phase"] = phase
                df_tmp["effect"] = effect
                df_tmp["alpha"] = alpha_list
                df_tmp["atlas"] = atlas
                D.append(df_tmp)
    
    return pd.concat(D)


def get_enc_ret_overlap_summ(subj = None, 
                             smooth = 3, 
                             subtract_mean = True, 
                             atlas_spaces = ["SUIT3", "fs32k"],
                             recall_dirs = [None], 
                             type = "CondHalf", 
                             ses_id = "ses-02", 
                             verbose = False):


    """
    """
    # get dataset class
    dset = ds.get_dataset_class(gl.base_dir, "WMFS")
    D = []
    for atlas in atlas_spaces:
        for dcall in recall_dirs:
            # get contrasts 
            ## data[0]: enc effect, data[1]: ret effect
            enc_data, ret_data = get_enc_ret_contrast(subj = subj, 
                                                       smooth = smooth, 
                                                       atlas_space = atlas,
                                                       type = type, 
                                                       recall_dir = dcall, 
                                                       ses_id = ses_id)

            
            
            # calculate reliabilities
            r_list = []
            for s in range(enc_data.shape[0]):
                enc_dat = np.nan_to_num(enc_data[s])
                ret_dat = np.nan_to_num(ret_data[s])
                # subtract_mean?
                if subtract_mean:
                    enc_dat = enc_dat - np.nanmean(enc_dat, axis = 1, keepdims=True)
                    ret_dat = ret_dat - np.nanmean(ret_dat, axis = 1, keepdims=True)

                r_list.append(corr_util.cosang(enc_dat,ret_dat))

            # making the summary dataframe
            ## get list of subjects
            subj_list = dset.get_participants().participant_id
            df_tmp = pd.DataFrame()
            df_tmp["sn"] = subj_list
            df_tmp["recall_dir"] = str(dcall)
            df_tmp["r"] = r_list
            df_tmp["atlas"] = atlas
            D.append(df_tmp)
    return pd.concat(D)


def get_dir_load_overlap_summ(subj = None, 
                          smooth = 3, 
                          subtract_mean = True, 
                          atlas_spaces = ["SUIT3", "fs32k"],
                          phases = ["enc", "ret"], 
                          type = "CondHalf", 
                          ses_id = "ses-02", 
                          verbose = False):

    """
    """
    # get dataset class
    dset = ds.get_dataset_class(gl.base_dir, "WMFS")
    D = []
    for atlas in atlas_spaces:
        for p, phase in enumerate(phases):
            # get contrasts 
            ## data[0]: dir effect, data[1]: load effect
            dir_data, load_data = get_dir_load_contrast(subj = subj, 
                                                        smooth = smooth, 
                                                        atlas_space = atlas,
                                                        type = type, 
                                                        phase = p, 
                                                        ses_id = ses_id)
            
            # calculate reliabilities
            r_list = []
            for s in range(dir_data.shape[0]):
                dir_dat = np.nan_to_num(dir_data[s])
                load_dat = np.nan_to_num(load_data[s])
                # subtract_mean?
                if subtract_mean:
                    dir_dat = dir_dat - np.nanmean(dir_dat, axis = 1, keepdims=True)
                    load_dat = load_dat - np.nanmean(load_dat, axis = 1, keepdims=True)

                r_list.append(corr_util.cosang(dir_dat,load_dat))

            # making the summary dataframe
            ## get list of subjects
            subj_list = dset.get_participants().participant_id
            df_tmp = pd.DataFrame()
            df_tmp["sn"] = subj_list
            df_tmp["phase"] = phase
            df_tmp["r"] = r_list
            df_tmp["atlas"] = atlas
            D.append(df_tmp)
    
    return pd.concat(D)


def get_enc_ret_rel_summ_depricated(subj = None, 
                                    smooth = 3, 
                                    subtract_mean = True, 
                                    atlas_spaces = ["SUIT3", "fs32k"],
                                    type = "CondHalf", 
                                    ses_id = "ses-02", 
                                    verbose = False):

    """
    """
    # get partitions
    partition = type[4:].lower()
    D = []
    for atlas in atlas_spaces:
        # get dataset
        data,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                        atlas=atlas,
                                        sess=ses_id,
                                        subj=subj,
                                        type = type,  
                                        smooth = smooth)

        for p, phase in enumerate(['enc', 'ret']):
            info_loop = info.copy()
            if verbose:
                print(f"- Doing {atlas} {p}")
            # get indices for the phase
            idx = info_loop.phase.values == p
            # get codition vector
            cond_vec = info_loop.cond_num.values
            cond_vec[np.logical_not(idx)] = 0
            # get within subject reliability
            r = ds.reliability_within_subj(data, 
                                            info_loop[partition].values, 
                                            cond_vec,
                                            voxel_wise=False,
                                            subtract_mean=subtract_mean)

            # the first column of r will be all nans
            # because the condition vector has a value for
            # the condition of interest and 0 everywhere else
            # discarding the nans 
            # make the dataframe
            df=pd.DataFrame()
            df["sn"] = dset.get_participants().participant_id
            df["atlas"] = atlas
            df["reliability"] = np.nanmean(r, axis = 1)
            df["contrast"] = phase

            D.append(df)
    return pd.concat(D)


def plot_rgb_map(data_rgb, 
                 atlas_space = "SUIT3", 
                 scale = [0.02, 1, 0.02], 
                 threshold = [0.02, 1, 0.02]):
    """
    plots rgb map of overlap on flatmap
    Args:
        data_rgb (np.ndarray) - 3*p array containinig rgb values per voxel/vertex
        atlas_space (str) - the atlas you are in, either SUIT3 or fs32k
        scale (list) - how much do you want to scale
        threshold (list) - threshold to be applied to the values
    Returns:
        ax (plt axes object) 
    """
    if atlas_space == "SUIT3":
        atlas, a_info = am.get_atlas(atlas_space,gl.atlas_dir)
        Nii = atlas.data_to_nifti(data_rgb)
        data = suit.vol_to_surf(Nii,space='SUIT')
        rgb = suit.flatmap.map_to_rgb(data,scale=scale,threshold=threshold)
        ax = suit.flatmap.plot(rgb,overlay_type='rgb', colorbar = True)
    elif atlas_space == "fs32k":
        # get the data into surface
        atlas, a_info = am.get_atlas('fs32k',gl.atlas_dir)
        
        dat_cifti = atlas.data_to_cifti(data_rgb)

        # get the lists of data for each hemi
        dat_list = nt.surf_from_cifti(dat_cifti)

        ax = []
        for i,hemi in enumerate(['L', 'R']):
            plt.figure()
            rgb = suit.flatmap.map_to_rgb(dat_list[i].T,scale,threshold=threshold)
            ax.append(sa.plot.plotmap(rgb, surf = f'fs32k_{hemi}',overlay_type='rgb'))

    return ax

if __name__=="__main__":
    D =get_enc_ret_overlap_summ(subj = None, 
                             smooth = 3, 
                             subtract_mean = True, 
                             atlas_spaces = ["SUIT3", "fs32k"],
                             recall_dirs = [None], 
                             type = "CondAll", 
                             ses_id = "ses-02", 
                             verbose = False)