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
import selective_recruitment.globals as gl
import selective_recruitment.recruite_ana as ra
import selective_recruitment.rsa as srsa
import cortico_cereb_connectivity.evaluation as ccev
import SUITPy as suit
# import surfAnalysisPy.stats as sas
import surfAnalysisPy as sa
import matplotlib.pyplot as plt 
from scipy.stats import norm, ttest_1samp

#
import nibabel as nb
from nilearn import plotting
import nitools as nt
import nitools.cifti as ntcifti
import SUITPy.flatmap as flatmap
from nilearn import plotting
import scipy.stats as ss

wkdir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'

# TODO: MDS plot in the condition space

def divide_by_horiz(atlas_space = "SUIT3", label = "NettekovenSym68c32"):
    """
    Create a mask that divides the cerebellum into anterior and posterior
    demarkated by horizontal fissure
    And then creates a new label file (alongside lookup table) with parcels divided by
    horizontal fissure
    Args:
        atlas_space (str) - string representing the atlas space
    Returns:
        mask_data (np.ndarray)
        mask_nii (nb.NiftiImage)
    """
    # create an instance of atlas object
    atlas_suit, ainfo = am.get_atlas(atlas_space, gl.atlas_dir)

    # load in the lobules parcellation
    lobule_file = f"{gl.atlas_dir}/tpl-SUIT/atl-Anatom_space-SUIT_dseg.nii"
    
    # get the lobules in the atlas space
    lobule_data, lobules = atlas_suit.get_parcel(lobule_file)

    # load the lut file for the lobules 
    idx_lobule, _, lobule_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-Anatom.lut")

    # demarcate the horizontal fissure
    ## horizontal fissure is between crusI and crusII.
    ## everything above crusII is the anterior part
    ## find the indices for crusII
    crusII_idx = [lobule_names.index(name) for name in lobule_names if "CrusII" in name]
    posterior_idx = idx_lobule[min(crusII_idx):]
    anterior_idx = idx_lobule[0:min(crusII_idx)]

    # assign value 1 to the anterior part
    anterior_mask = np.isin(lobule_data, anterior_idx)

    # assign value 2 to the posterior part
    posterior_mask = np.isin(lobule_data, posterior_idx)

    # get the label file
    lable_file = f"{gl.atlas_dir}/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii"
    # load the lut file for the label  
    idx_label, colors, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")

    # get the parcels in the atlas space
    label_data, labels = atlas_suit.get_parcel(lable_file)

    # loop over regions and divide them into two parts
    idx_new = [0]
    colors_new = [[0, 0, 0, 1]]
    label_new = ["0"] 
    label_array = np.zeros(label_data.shape)
    idx_num = 1
    for i in labels:

        # get a copy of label data
        label_copy = label_data.copy()

        # convert all the labels other than the current one to NaNs
        label_copy[label_copy != i] = 0

        # get the anterior and posterior part
        label_anterior = label_copy * anterior_mask
        label_posterior = label_copy * posterior_mask

        if any(label_anterior): # some labels only have posterior parts
            # get the anterior part
            label_new.append(f"{label_names[i]}_A")
            idx_new.append(idx_num)
            colors_new.append(list(np.append(colors[i, :], 1))) 
            label_array[np.argwhere(label_anterior)] = idx_num
            idx_num=idx_num+1

        if any(label_posterior):
            # get the posterior part
            label_new.append(f"{label_names[i]}_P")
            idx_new.append(idx_num)
            colors_new.append(list(np.append(colors[i, :], 1))) 
            label_array[np.argwhere(label_posterior)] = idx_num

            idx_num=idx_num+1


    # create a nifti object
    nii = atlas_suit.data_to_nifti(label_array)

    # save the nifti
    nb.save(nii, f"{gl.atlas_dir}/tpl-SUIT/atl-{label}AP_space-SUIT_dseg.nii")

    # save the lookuptable
    nt.save_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}AP.lut", idx_new, np.array(colors_new), label_new)
    return nii

def calculate_R(A, B):
    SAB = np.nansum(A * B, axis=0)
    SBB = np.nansum(B * B, axis=0)
    SAA = np.nansum(A **2, axis=0)  # use np.nanmean(Y) here?

    R = np.nansum(SAB) / np.sqrt(np.nansum(SBB) * np.nansum(SAA))
    return R

def smooth_cifti_data(surface_sigma = 3, 
                      volume_sigma = 3,
                      atlas_space = "fs32k",
                      type = "CondAll", 
                      ses_id = "ses-02"):
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")
    # get list of subjects
    T = Data.get_participants()
    subject_list = list(T.participant_id.values)
    subject_list.append("group")

    # get the surfaces for smoothing
    surfs = [Data.atlas_dir + f'/tpl-fs32k/tpl_fs32k_hemi-{h}_inflated.surf.gii' for i, h in enumerate(['L', 'R'])]
    # loop over subjects, load data, smooth and save with s prefix
    for s, subject in enumerate(subject_list):
        # load the cifti
        cifti_fpath = Data.data_dir.format(subject)
        cifti_fname = f"{subject}_space-{atlas_space}_{ses_id}_{type}.dscalar.nii"
        cifti_input = f"{cifti_fpath}/{cifti_fname}"
        cifti_output = f"{cifti_fpath}/s{cifti_fname}"

        # smooth the file
        ntcifti.smooth_cifti(cifti_input, 
                              cifti_output,
                              surfs[0], 
                              surfs[1], 
                              surface_sigma = surface_sigma, 
                              volume_sigma = volume_sigma, 
                              direction = "COLUMN", 
                              )
    return

def calc_contrast(data, info):
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
    # load effect will be calculated on the forwards conditions only
    ## get the index for baseline condition
    idx_base_load = (info.load == 2) & (info.recall == 1)
    ## get the index for effect condition
    idx_effect_load = (info.load == 6) & (info.recall == 1)
    ## load effect
    data_load = np.nanmean(data[idx_effect_load, :], axis = 0) - np.nanmean(data[idx_base_load, :], axis = 0)
    
    # calculate the effect of recall direction
    ## get the index for baseline condition
    idx_base_recall = (info.recall == 1) 
    ## get the index for effect condition
    idx_effect_recall = (info.recall == 0)
    ## recall effect
    data_recall = np.nanmean(data[idx_effect_recall, :], axis = 0) - np.nanmean(data[idx_base_recall, :], axis = 0)
    return data_load, data_recall

def load_contrast(ses_id = 'ses-02',
                subj = "group",atlas_space='SUIT3',
                phase=0, 
                type = "CondAll",
                smooth = True, 
                verbose = False):
    """
    1. Gets group data 
    2. Calculates the desired contrast (uses the info file to get conditions)
    """    
    data,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type,  
                                    smooth = smooth, 
                                    verbose = verbose)
    data = data[0]

    
    idx = (info.phase == phase)
        
    info = info.loc[idx]
    data = data[idx, :]
    
    # get contrasts
    data_load, data_recall = calc_contrast(data, info)
    return data_load,data_recall

def calc_overlap_corr(atlas_space = "fs32k", 
                      type = "CondAll", 
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

            # calculate correlation between the two maps
            R = calculate_R(data_load, data_recall)

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
            col_names.append(f"{phase}_recall")

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

def calc_effect_reliability(atlas_space = "fs32k", 
                            smooth = True, verbose = False):
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")

    # create an atlas object 
    atlas, _ = am.get_atlas(atlas_space, Data.atlas_dir)
    D = []
    for subject in Data.get_participants().participant_id:
        
        for p, phase in enumerate(['Enc', 'Ret']):
            data,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas=atlas_space,
                                    sess="ses-02",
                                    subj=subject,
                                    type = "CondHalf",  
                                    smooth = smooth, 
                                    verbose = verbose)
            load_list = []
            recall_list = []
            for half in [1, 2]:
                idx_p = (info.phase == p)
                idx_h = (info.half == half)
                idx = idx_p & idx_h

                info_half = info.loc[idx]
                data_half = data[0, idx, :]
                # get data for the half
                load_half, recall_half = calc_contrast(data_half, info_half)
                load_list.append(load_half)
                recall_list.append(recall_half)

            effect_list = []
            effect_list.append(load_list)
            effect_list.append(recall_list)
            
            # get split half reliability
            for eff, effect in enumerate(['load', 'recall']):
                R = calculate_R(effect_list[eff][0], effect_list[eff][1])
                # get info
                R_dict = {}
                R_dict["dataset"] = "WMFS"
                R_dict["ses_id"] ="ses-02"
                R_dict["phase"] = phase
                R_dict["effect"] = effect
                R_dict["atlas"] = atlas_space
                R_dict["R"] = R
                R_dict["sn"] = subject

                R_df = pd.DataFrame(R_dict, index = [0])
                D.append(R_df)

    return pd.concat(D)

def get_overlap_summary(type = "CondAll", smooth = True, verbose = False):
    """
    wrapper to get the dataframe for quantifying the overlap between the load and recall effects
    """

    D_fs = calc_overlap_corr(atlas_space = "fs32k", 
                            type = type, 
                            smooth = smooth, 
                            verbose=verbose,
                            save = False)
    D_suit = calc_overlap_corr(atlas_space = "SUIT3", 
                            type = type, 
                            smooth = smooth, 
                            verbose=verbose, 
                            save = False)
    
    # concatenate the two dataframes to return a full
    D = pd.concat([D_fs, D_suit], axis = 0)

    return D

def get_reliability_summary(smooth = True, verbose = False):
    """
    wrapper to get the dataframe for reliability of load and recall effects
    """
    D_fs = calc_effect_reliability(atlas_space = "fs32k", 
                                   verbose = verbose,
                                   smooth = smooth)
    D_suit = calc_effect_reliability(atlas_space = "SUIT3", verbose=verbose, 
                                    smooth = smooth)

    # concatenate the two dataframes to return a full
    D = pd.concat([D_fs, D_suit], axis = 0)
    return D

def plot_overlap_cerebellum(subject = "group", 
                            phase = 0,
                            smooth = False, 
                            scale = None, 
                            type = 'CondAll',
                            save_svg = False, 
                            verbose = False, 
                            threshold = None):
    """
    Makes an overlap plot for the cerebellum 
    """
    load_eff,dir_eff=load_contrast(ses_id = 'ses-02',
                                   subj = subject,
                                   atlas_space='SUIT3',
                                   phase=phase, 
                                   verbose = verbose, 
                                   smooth = False   , 
                                   type = type)
    data=np.c_[dir_eff,
               np.zeros(load_eff.shape),
               load_eff].T # Leave the green gun empty 
    atlas, a_info = am.get_atlas('SUIT3',gl.atlas_dir)
    Nii = atlas.data_to_nifti(data)
    data = suit.vol_to_surf(Nii,space='SUIT')
    rgb = suit.flatmap.map_to_rgb(data,scale=scale,threshold=threshold)
    ax = suit.flatmap.plot(rgb,overlay_type='rgb', colorbar = True)
    return ax

def plot_overlap_cortex(subject = "group", 
                        phase=0, 
                        smooth = True, 
                        scale = None,
                        type = "CondAll",
                        verbose = False, 
                        threshold = None):
    """
    Makes an overlap plot for the cerebellum 
    """
    load_eff,dir_eff= load_contrast(ses_id = 'ses-02',
                                    subj = subject,
                                    atlas_space='fs32k',
                                    phase=phase, 
                                    type = type, 
                                    verbose = verbose,
                                    smooth = smooth)
    # get the data into surface
    atlas, a_info = am.get_atlas('fs32k',gl.atlas_dir)
    load_cifti = atlas.data_to_cifti(load_eff.reshape(-1, 1).T)
    dir_cifti = atlas.data_to_cifti(dir_eff.reshape(-1, 1).T)

    # get the lists of data for each hemi
    load_list = nt.surf_from_cifti(load_cifti)
    dir_list = nt.surf_from_cifti(dir_cifti)

    ax = []
    for i,hemi in enumerate(['L', 'R']):
        # plt.figure()
        data=np.c_[dir_list[i].T,
                np.zeros(load_list[i].T.shape),
                load_list[i].T] # Leave the green gun empty 

        # plt.subplot(1,2,i+1)
        rgb = suit.flatmap.map_to_rgb(data,scale,threshold=threshold)
        ax.append(sa.plot.plotmap(rgb, surf = f'fs32k_{hemi}',overlay_type='rgb'))
    return ax

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
    load_sub_list = nt.surf_from_cifti(atlas.data_to_cifti(data_load_arr))
    recall_sub_list = nt.surf_from_cifti(atlas.data_to_cifti(data_recall_arr))
    
    t_val_load_list = []
    t_val_recall_list = []
    for h, hemi in enumerate(["L", "R"]):

        t_val_load, p_val_load = ttest_1samp(load_sub_list[h], axis = 0, popmean = 0, nan_policy = 'omit', alternative = 'greater')
        t_val_load_list.append(t_val_load)

        t_val_recall, p_val_recall = ttest_1samp(recall_sub_list[h], axis = 0, popmean = 0, nan_policy = 'omit', alternative='greater')
        t_val_recall_list.append(t_val_recall)

    return t_val_load_list, t_val_recall_list

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

def plot_contrast_cortex(subject, phase, effect, smooth = True, save_svg = False):
    """
    """    

    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")
    
    # surfaces for plotting
    surfs = [Data.atlas_dir + f'/tpl-fs32k/tpl_fs32k_hemi-{h}_inflated.surf.gii' for i, h in enumerate(['L', 'R'])]
    # get the cerebellar atlas object
    atlas, ainfo = am.get_atlas('SUIT3', Data.atlas_dir)
    # which contrast?
    effect_name = f"{phase}_{effect}"
    # load the cifti
    if smooth:
        cifti = nb.load(f"{wkdir}/data/{subject}/sload_recall_space_fs32k_CondAll_{subject}.dscalar.nii")
    else: 
        cifti = nb.load(f"{wkdir}/data/{subject}/load_recall_space_fs32k_CondAll_{subject}.dscalar.nii")
    # get the names of the contrasts
    con_names = list(cifti.header.get_axis(0).name)

    # print(cifti.get_fdata().shape)

    # get the index of the contrast in question
    idx = con_names.index(effect_name)
    dat_list = nt.surf_from_cifti(cifti)

    # get the numpy array corresponding to the contrast
    img_con_list = [dat_list[i][idx, :].reshape(-1, 1) for i, h in enumerate(['L', 'R'])]

    

    fig = plotting.plot_surf_stat_map(
                                        surfs[0], img_con_list[0], hemi='left',
                                        # title='Surface left hemisphere',
                                        colorbar=True, 
                                        view = 'lateral',
                                        cmap="coolwarm",
                                        engine='plotly',
                                        title = f'{phase}_{effect}_{subject}',
                                        symmetric_cbar = True,
                                        vmax = 0.1
                                    )

    ax = fig.figure
    if save_svg:
        ax.write_image(f"{wkdir}/figures/cortex_left_{phase}_{effect}_{subject}.svg")
    return ax

if __name__=="__main__":
    divide_by_horiz()
    