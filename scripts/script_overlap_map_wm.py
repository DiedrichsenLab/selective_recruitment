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
import cortico_cereb_connectivity.evaluation as ccev
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

def calc_contrast(X, contrast_vector):
    """
    Calculates contrast of interest 
    """
    return contrast_vector.T@X

def get_enc_ret_contrast(subj = "group", 
                        smooth = 3, 
                        atlas_space = "SUIT3",
                        type = "CondAll", 
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
    
    # make contrast vector for encoding and retrieval
    # get indices for all the enc contrasts
    idx_enc = info.phase == 0
    c_enc = (idx_enc.values*1)/np.sum(idx_enc.values*1)
    enc_data = calc_contrast(data[0, :, :], c_enc)
    # get indices for all retrieval contrasts
    idx_ret = info.phase == 1
    c_ret = (idx_ret.values*1)/np.sum(idx_ret.values*1)
    ret_data = calc_contrast(data[0, :, :], c_ret)

    # prepare for rgb map
    dat_rgb =np.c_[enc_data,
               np.zeros(enc_data.shape),
               ret_data].T # Leave the green gun empty 

    return dat_rgb

def get_load_dir_contrast(subj = "group", 
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
    
    # make contrast vector for load effect
    idx_load6 = (info.phase == phase) & (info.recall == 1) & (info.load == 6)
    idx_load2 = (info.phase == phase) & (info.recall == 1) & (info.load == 2)
    c_load = ((idx_load6.values*1)/np.sum(idx_load6.values*1)) - ((idx_load2.values*1)/np.sum(idx_load2.values*1))
    
    # make contrast vector for dir effect
    idx_bw = (info.phase == phase) & (info.recall == 0) 
    idx_fw = (info.phase == phase) & (info.recall == 1) 
    c_dir = ((idx_bw.values*1)/np.sum(idx_bw.values*1)) - ((idx_fw.values*1)/np.sum(idx_fw.values*1))
    
    dir_data = calc_contrast(data[0, :, :], c_dir)
    load_data = calc_contrast(data[0, :, :], c_load)

    # prepare for rgb map
    dat_rgb =np.c_[dir_data,
               np.zeros(dir_data.shape),
               load_data].T # Leave the green gun empty 
    return dat_rgb

def get_enc_ret_dir_contrast(subj = "group", 
                            smooth = 3, 
                            atlas_space = "SUIT3",
                            type = "CondAll",
                            dir = 0,  
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
    
    # make contrast vector for encoding and retrieval
    # get indices for all the enc contrasts
    idx_enc = (info.phase == 0) & (info.recall == dir)
    c_enc = (idx_enc.values*1)/np.sum(idx_enc.values*1)
    enc_data = calc_contrast(data[0, :, :], c_enc)
    # get indices for all retrieval contrasts
    idx_ret = info.phase == 1 & (info.recall == dir)
    c_ret = (idx_ret.values*1)/np.sum(idx_ret.values*1)
    ret_data = calc_contrast(data[0, :, :], c_ret)

    # prepare for rgb map
    dat_rgb =np.c_[enc_data,
               np.zeros(enc_data.shape),
               ret_data].T # Leave the green gun empty 
    return

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


def calc_overlap_corr(atlas_space = "fs32k", 
                      type = "CondAll",
                      subtract_mean = False,  
                      smooth = True,
                      group = False, 
                      verbose = False,
                      save = False):
    """
    calculates the correlations between effects
    """
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")
    if group:
        subject_list = ["group"]
    else: 
        subject_list = Data.get_participants().participant_id
    # create an atlas object 
    atlas, _ = am.get_atlas(atlas_space, Data.atlas_dir)
    D = []
    for subject in subject_list:
        effect_list = []
        col_names = []
        for p, phase in enumerate(['Enc', 'Ret']):
            data_load, data_recall = load_contrast(ses_id = 'ses-02',
                                                    subj = subject, atlas_space=atlas_space,
                                                    phase = p,
                                                    type = type, 
                                                    verbose = verbose, 
                                                    smooth = smooth)

            # Remove the mean per voxel before correlation calc?
            if subtract_mean:
                data_load -= np.nanmean(data_load, axis=0)
                data_recall -= np.nanmean(data_recall, axis=0)

            # calculate correlation between the two maps
            ## nan to zero
            data_load = np.nan_to_num(data_load)
            data_recall = np.nan_to_num(data_recall)

            ## cosine angle between the two maps
            R = corr_util.cosang(data_load,data_recall)

            # get info
            R_dict = {}
            R_dict["dataset"] = "WMFS"
            R_dict["ses_id"] ="ses-02"
            R_dict["phase"] = phase
            R_dict["atlas"] = atlas_space
            R_dict["R"] = R
            R_dict["sn"] = subject

            R_df = pd.DataFrame(R_dict, index = [0])
            D.append(R_df)

            # prepare data to be saved as cifti
            effect_list.append(data_load.reshape(-1, 1).T)
            col_names.append(f"{phase}_load")

            effect_list.append(data_recall.reshape(-1, 1).T)
            col_names.append(f"{phase}_recall_dir")

        # make ciftis and save
        data_effect = np.concatenate(effect_list, axis = 0)
        # save ciftis as dscalar
        effect_obj = atlas.data_to_cifti(data_effect, row_axis = col_names)

        # save the cifti file
        if save:
            save_path = f"{wkdir}/data/{subject}"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            nb.save(effect_obj, f"{save_path}/load_recall_space_{atlas_space}_{type}_{subject}.dscalar.nii")        
    return pd.concat(D)

def calc_effect_reliability(atlas_space = "fs32k", subtract_mean = True, 
                            smooth = True, verbose = False):
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")

    # create an atlas object 
    atlas, _ = am.get_atlas(atlas_space, Data.atlas_dir)
    D = []
    for ss, subject in enumerate(Data.get_participants().participant_id):

        data,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas=atlas_space,
                                    sess="ses-02",
                                    subj=subject,
                                    type = "CondHalf",  
                                    smooth = smooth, 
                                    verbose = verbose)
        for p, phase in enumerate(['Enc', 'Ret']):

            load_half = []
            recall_dir_half = []
            for half in [1, 2]:
                # get indices for the current phase
                idx_p = (info.phase == p)
                
                # get info and data for the half
                idx_h = info.half == half
                info_half = info.loc[idx_h*idx_p]
                data_half = data[0, idx_h*idx_p, :]
                # calculate the contrasts
                load, recall_dir = calc_contrast(data_half, info_half)
                # Remove the mean per voxel before correlation calc?
                if subtract_mean:
                    load -= np.nanmean(load, axis=0)
                    recall_dir -= np.nanmean(recall_dir, axis=0)

                load = np.nan_to_num(load)
                recall_dir = np.nan_to_num(recall_dir)

                load_half.append(load)
                recall_dir_half.append(recall_dir)

            # calculating reliabilities per effect per phase
            ## first get the load and recall of halves into a np.ndarray
            load_arr = np.c_[load_half[0], load_half[1]]
            recall_dir_arr = np.c_[recall_dir_half[0], recall_dir_half[1]]
            ## append to a list containing effects
            effect_list = [] # will contain two halves of data for each effect. load comes first
            effect_list.append(load_arr)
            effect_list.append(recall_dir_arr)
            ## keep in mind that load comes first
            for e, effect in enumerate(["load", "recall_dir"]):
                alpha_cron = corr_util.calc_cronbach(effect_list[e].T)

                R = corr_util.cosang(effect_list[e][:, 0], effect_list[e][:, 1])

                # preparing the dataframe
                R_dict = {}
                R_dict["dataset"] = "WMFS"
                R_dict["ses_id"] ="ses-02"
                R_dict["phase"] = phase
                R_dict["effect"] = effect
                R_dict["atlas"] = atlas_space
                R_dict["R"] = R
                R_dict["a_cron"] = alpha_cron
                R_dict["sn"] = subject


                R_df = pd.DataFrame(R_dict, index = [ss])
                D.append(R_df)

    return pd.concat(D)

def get_overlap_summary(type = "CondAll", subtract_mean = False, smooth = True, verbose = False):
    """
    wrapper to get the dataframe for quantifying the overlap between the load and recall effects
    """

    D_fs = calc_overlap_corr(atlas_space = "fs32k", 
                            type = type, 
                            subtract_mean=subtract_mean, 
                            smooth = smooth, 
                            verbose=verbose,
                            save = False)
    D_suit = calc_overlap_corr(atlas_space = "SUIT3", 
                            type = type, 
                            subtract_mean=subtract_mean,
                            smooth = smooth, 
                            verbose=verbose, 
                            save = False)
    
    # concatenate the two dataframes to return a full
    D = pd.concat([D_fs, D_suit], axis = 0)

    return D

def get_effect_reliability_summary(smooth = True, verbose = False, subtract_mean = True):
    """
    wrapper to get the dataframe for reliability of load and recall effects
    """
    D_fs = calc_effect_reliability(atlas_space = "fs32k", 
                                   subtract_mean=subtract_mean,
                                   verbose = verbose,
                                   smooth = smooth)
    D_suit = calc_effect_reliability(atlas_space = "SUIT3", 
                                     subtract_mean=subtract_mean, 
                                     verbose=verbose, 
                                     smooth = smooth)

    # concatenate the two dataframes to return a full
    D = pd.concat([D_fs, D_suit], axis = 0)
    return D

def conjunction_ana_cortex(type = "CondAll", 
                           phase = 0,
                           atlas_space = "fs32k", 
                           smooth = True):

    """
    """
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")
    # get list of subjects
    T = Data.get_participants()
    subject_list = list(T.participant_id.values)

    # get the atlas object for cifit/nifti creation
    atlas, ainfo = am.get_atlas("fs32k", Data.atlas_dir)

    # loop over subjects and load the cifti
    data_load_subs = []
    data_recall_subs = []
    for s, subject in enumerate(subject_list):

        data_load, data_recall = load_contrast(ses_id = 'ses-02',
                                    subj = subject,atlas_space=atlas_space,
                                    phase=phase, 
                                    type = type,
                                    smooth = smooth, 
                                    verbose = False)

        # get data for load
        data_load_subs.append(data_load[np.newaxis, ...])

        # get the data for recall
        data_recall_subs.append(data_recall[np.newaxis, ...])

    # create array containing data for all subjects
    data_load_arr = np.concatenate(data_load_subs, axis = 0)
    data_recall_arr = np.concatenate(data_recall_subs, axis = 0)

    # do the test per hemisphere
    # load_sub_list = nt.surf_from_cifti(atlas.data_to_cifti(data_load_arr))
    # recall_sub_list = nt.surf_from_cifti(atlas.data_to_cifti(data_recall_arr))
    
    # t_val_load_list = []
    # t_val_recall_list = []
    # for h, hemi in enumerate(["L", "R"]):

    t_val_load, p_val_load = ttest_1samp(data_load_arr, axis = 0, popmean = 0, nan_policy = 'omit', alternative = 'greater')
    # t_val_load_list.append(t_val_load)
    load_list = nt.surf_from_cifti(atlas.data_to_cifti(t_val_load))

    t_val_recall, p_val_recall = ttest_1samp(data_recall_arr, axis = 0, popmean = 0, nan_policy = 'omit', alternative='greater')
    # t_val_recall_list.append(t_val_recall)

    # return t_val_load_list, t_val_recall_list
    return t_val_load, t_val_recall

def conjunction_ana_cerebellum(type = "CondAll", 
                           phase = 0,
                           atlas_space = "SUIT3", 
                           smooth = True):

    """
    """
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")
    # get list of subjects
    T = Data.get_participants()
    subject_list = list(T.participant_id.values)

    # get the atlas object for cifit/nifti creation
    atlas, ainfo = am.get_atlas(atlas_space, Data.atlas_dir)

    # loop over subjects and load the cifti
    data_load_subs = []
    data_recall_subs = []
    for s, subject in enumerate(subject_list):

        data_load, data_recall = load_contrast(ses_id = 'ses-02',
                                    subj = subject,atlas_space=atlas_space,
                                    phase=phase, 
                                    type = type,
                                    smooth = smooth, 
                                    verbose = False)

        # get data for load
        data_load_subs.append(data_load[np.newaxis, ...])

        # get the data for recall
        data_recall_subs.append(data_recall[np.newaxis, ...])

    # create array containing data for all subjects
    data_load_arr = np.concatenate(data_load_subs, axis = 0)
    data_recall_arr = np.concatenate(data_recall_subs, axis = 0)

    # do test
    t_val_load, p_val_load = ttest_1samp(data_load_arr, axis = 0, popmean = 0, nan_policy='omit', alternative="greater")

    t_val_recall, p_val_recall = ttest_1samp(data_recall_arr, axis = 0, popmean = 0, nan_policy = 'omit', alternative = "greater")

    return t_val_load, t_val_recall

def plot_contrast_cerebellum(subject, phase = "Enc", effect = "load", save_svg = False):
    """
    """
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")

    # get the cerebellar atlas object
    atlas, ainfo = am.get_atlas('SUIT3', Data.atlas_dir)
    # which contrast?
    effect_name = f"{phase}_{effect}"
    # load the cifti
    cifti = nb.load(f"{wkdir}/data/{subject}/load_recall_space_SUIT3_CondAll_{subject}.dscalar.nii")

    # get the names of the contrasts
    con_names = list(cifti.header.get_axis(0).name)

    # print(cifti.get_fdata().shape)

    # get the index of the contrast in question
    idx = con_names.index(effect_name)

    # get the map
    cerebellar_map = cifti.get_fdata()[idx, :]

    # make a nifti object of the map
    nifti = atlas.data_to_nifti(cerebellar_map)

    # transfer to flat surface
    img_flat = suit.flatmap.vol_to_surf([nifti], stats='nanmean', space = 'SUIT')

    # plot
    ax = suit.flatmap.plot(data=img_flat, 
                      render="plotly", 
                      hover='auto', 
                      cmap = "coolwarm", 
                      colorbar = True, 
                      bordersize = 1, 
                      cscale = (-0.1, 0.1))

    # ax.show()
    ax.update_layout(title = {'text':f"{phase}_{effect}_{subject}", 
                              'y':0.95,
                              'x':0.5,
                              'xanchor': 'center'})
    ax.show()
    if save_svg:
        ax.write_image(f"{wkdir}/figures/flatmap_{phase}_{effect}_{subject}.svg")
    return 


if __name__=="__main__":
    
    pass