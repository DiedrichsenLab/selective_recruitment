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
from PcmPy.matrix import indicator
from scipy.linalg import solve, pinv
from scipy.spatial import procrustes
from numpy.linalg import eigh

wkdir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'

# TODO: MDS plot in the condition space


def calc_contrast(X, info, phase = 0):
    """
    calculates load and recall contrasts of interest in the phase
    load is defined as the difference between load 6 and load 2 in forward recall
    recall dir effect is defined as the difference between backwards and forwards
    Args:
        X (np.ndarray) - n_subj*n_cond*n_vox
        info (pd.DataFrame) - pandas dataframe
        phase (int) - 0 for encoding, 1 for retrieval
    Returns:
        X_con (np.ndarray) - n_subj*2*n_vox

    """
    n_cond = X.shape[0]
    n_vox = X.shape[1]
    
    # get the array for load contrast
    # NOTE: load contrast is only calculated for forwards conditions
    load_effect = ((info.phase == phase) & (info.recall == 1) & (info.load == 6))*1
    load_base = ((info.phase == phase) & (info.recall == 1) &(info.load == 2))*1
    c_load = load_effect/np.sum(load_effect) - load_base/np.sum(load_base)
        
    # get the array for recall direction contrast
    dir_effect = ((info.phase == phase) & (info.recall == 1))*1 
    dir_base = ((info.phase == phase) & (info.recall == 0))*1
    c_dir = dir_effect/np.sum(dir_effect) - dir_base/np.sum(dir_base)

    # initializing the effect array
    # there are 2 effects of interest:
    # load: only in forward recall
    # recall_dir 
    # the first one is going to be the load (column 0)
    # the second one is going to be recall direction effect (column 1)
    X_con = np.zeros([n_vox, 2])
    # get load effect
    X_con[:, 0] = (c_load@X)
    # get recall effect 
    X_con[:, 1] = c_dir@X
    return X_con


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
    
    # get contrasts
    data_con = calc_contrast(data, info, phase=phase)
    return data_con, 


def calc_reliability(data,
                     info, 
                     partition = "run", 
                     phase = 0,  
                     subtract_mean = True):
    """
    calculates reliability
    """
    # get numbers 
    n_subj = data.shape[0]
    part_vec = info[partition]
    n_part = len(np.unique(part_vec))
    parts = np.unique(part_vec)

    r = np.zeros((n_subj, 2, n_part)) # correlation
    for s in np.arange(n_subj):
        for pn, part in enumerate(parts):
            i1 = part_vec == part
            i2 = part_vec != part
            # get info and data for the partitions
            X1 = data[s, i1, :]
            # get contrast for first partition
            data1 = calc_contrast(X1, info.loc[i1], phase=phase)
            X2 = data[s, i2, :]
            data2 = calc_contrast(X2, info.loc[i2], phase=phase)

            if subtract_mean:
                data1 -= np.nanmean(data1, axis=0)
                data2 -= np.nanmean(data2, axis=0)

            r[s, :, pn] = np.nansum(data1 * data2, axis = 0) / \
                np.sqrt(np.nansum(data1 * data1, axis = 0) * np.nansum(data2 * data2, axis = 0))
    return r


def calc_overlap(data, 
                 info, 
                 partition = "run",
                 phase = 0,  
                 subtract_mean = True 
                 ):
    """
    calculates reliability
    """
    # get numbers 
    n_subj = data.shape[0]
    part_vec = info[partition]
    n_part = len(np.unique(part_vec))
    parts = np.unique(part_vec)

    r = np.zeros((n_subj, n_part))
    for s in np.arange(n_subj):
        for pn, part in enumerate(parts):
            i1 = part_vec == part
            i2 = part_vec != part
            # get info and data for the partitions
            X1 = data[s, i1, :]
            # get contrast for first partition
            data1 = calc_contrast(X1, info.loc[i1], phase=phase)
            X2 = data[s, i2, :]
            data2 = calc_contrast(X2, info.loc[i2], phase=phase)

            if subtract_mean:
                data1 -= np.nanmean(data1, axis=0)
                data2 -= np.nanmean(data2, axis=0)

            data1 = np.nan_to_num(data1)
            data2 = np.nan_to_num(data2)

            r[s, pn] = corr_util.cosang(data1,data2)
    
    return r


def get_summary_reliability(type = "CondRun", 
                            partition = "run",
                            subtract_mean = True, 
                            smooth = False 
                            ):
    """
    """
    # loop over atlases
    df_list = []
    for atlas in ["SUIT3", "fs32k"]:
        data,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                        atlas=atlas,
                                        sess="ses-02",
                                        subj=None,
                                        type = type,  
                                        smooth = smooth, 
                                        verbose = False)

        # loop over phases 
        for p, phase in enumerate(["enc", "ret"]):
            r = calc_reliability(data,
                             info,
                             partition = partition,  
                             phase = p,  
                             subtract_mean = subtract_mean)

            # calc mean over partitions
            r_overal = np.nanmean(r, axis = 2)

            # make the summary df
            for i, effect in enumerate(["load", "dir"]):
                D = pd.DataFrame(r_overal[:, i], columns=["R"])
                D["effect"] = effect
                D["phase"] = phase
                D["atlas"] = atlas
                D["sn"] = dset.get_participants().participant_id

                # append to the list of datafraSmes
                df_list.append(D)
    return pd.concat(df_list, ignore_index=True)


def get_summary_overlap(type = "CondRun", 
                        partition = "run",
                        subtract_mean = True, 
                        smooth = False 
                        ):
    """
    """
    # loop over atlases
    df_list = []
    for atlas in ["SUIT3", "fs32k"]:
        data,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                        atlas=atlas,
                                        sess="ses-02",
                                        subj=None,
                                        type = type,  
                                        smooth = smooth, 
                                        verbose = False)

        # loop over phases 
        for p, phase in enumerate(["enc", "ret"]):
            r = calc_overlap(data,
                                 info,
                                 partition = partition,  
                                 phase = p,  
                                 subtract_mean = subtract_mean)

            # calc mean over partitions
            r_overal = np.nanmean(r, axis = 1)

            # make the summary df
            D = pd.DataFrame(r_overal, columns=["R"])
            D["phase"] = phase
            D["atlas"] = atlas
            D["sn"] = dset.get_participants().participant_id

            # append to the list of datafraSmes
            df_list.append(D)
    return pd.concat(df_list, ignore_index=True)


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


def est_G(Y, X=None, S=None):
    """
    Obtains a crossvalidated estimate of G
    Y = Z @ U + X @ B + E, where var(U) = G

    Parameters:
        Y (numpy.ndarray)
            Activity data
        Z (numpy.ndarray)
            2-d: Design matrix for conditions / features U
            1-d: condition vector
        part_vec (numpy.ndarray)
            Vector indicating the partition number
        X (numpy.ndarray)
            Fixed effects to be removed
        S (numpy.ndarray)

    Returns:
        G_hat (numpy.ndarray)
            n_cond x n_cond matrix
        Sig (numpy.ndarray)
            n_cond x n_cond noise estimate per block

    """

    n_channel , n_cond = Y.shape
    G = Y @ Y.T 

    return G


def get_region_info(label = 'NettekovenSym68c32AP'):
    # get the roi numbers of Ds only
    idx_label, colors, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")
    
    
    D_indx = [label_names.index(name) for name in label_names if "D" in name]
    D_name = [name for name in label_names if "D" in name]

    # get the colors of Ds
    colors_D = colors[D_indx, :]

    D_list = []
    for rr, reg in enumerate(D_indx):
        reg_dict = {}
        reg_dict['region'] = reg
        reg_dict['region_name'] = D_name[rr]
        reg_dict['region_id'] = rr
        if 'L' in label_names[reg]:
            reg_dict['side'] = 'L'
        if 'R' in label_names[reg]:
            reg_dict['side'] = 'R'
        if 'A' in label_names[reg]:
            reg_dict['anterior'] = True
        if 'P' in label_names[reg]:
            reg_dict['anterior'] =False
        D_list.append(pd.DataFrame(reg_dict, index=[rr]))
    Dinfo = pd.concat(D_list)

    return Dinfo, D_indx, colors_D


def calc_G_group(center = False, reorder = ['side', 'anterior']):
    """
    creating MDS plots with conditions as axes
    """
    # TODO: Goal: To investigate the activity profiles of different regions. In other words, how similar those regions are.
    # TODO: Get the similarity between regions. Use the variance covariance between activity profiles of regions wthin subjects
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")

    # get the data tensor
    tensor, info, _ = ds.get_dataset(gl.base_dir,"WMFS",atlas="SUIT3",sess="ses-02",type='CondAll', info_only=False)

    # create atlas object
    atlas_suit, _ = am.get_atlas("SUIT3",gl.atlas_dir)

    # make the label name for cerebellar parcellation
    lable_file = f"{gl.atlas_dir}/tpl-SUIT/atl-NettekovenSym68c32AP_space-SUIT_dseg.nii"

    # get info for D regions
    Dinfo, D_indx, colors_D = get_region_info(label = 'NettekovenSym68c32AP')
    # get parcels in the atlas

    label_vec, labels = atlas_suit.get_parcel(lable_file)
    # average data within parcels
    ## dimensions will be #subjects-by-#numberCondition-by-#region
    parcel_data, labels = ds.agg_parcels(tensor, label_vec, fcn=np.nanmean) 

    # get the parcel data corresponding to Ds
    parcel_data = parcel_data[:, :, D_indx]

    # get different dimensions of the data
    n_subj = parcel_data.shape[0]
    n_cond = parcel_data.shape[1]
    n_region = parcel_data.shape[2]
    # calculate average across subject
    data_final = np.nanmean(parcel_data, axis = 0)
    G = est_G(data_final.T)
    # pcm.plot_Gs(G[np.newaxis, ...],grid = None, labels=None)

    W, Glam = pcm.classical_mds(G,contrast=None,align=None,thres=0)

    Ginf=Dinfo.copy()
    if reorder:
        Ginf=Ginf.sort_values(reorder)
        ind=Ginf.index.to_numpy()
        G=G[ind,:][:,ind]
        Ginf=Ginf.reset_index()
    

    return G, W, Glam, Ginf, colors_D


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
        plt.figure()
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
    pass


    