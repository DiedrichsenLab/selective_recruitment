#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to make label files for the cortical and cerebellar rois

Created on 07/03/2023
Author: Ladan Shahshahani
"""
# import packages
import enum
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
from scipy.stats import norm, ttest_1samp

#
import os
import nibabel as nb
from nilearn import plotting
import nitools as nt
import SUITPy.flatmap as flatmap
from nilearn import plotting
import scipy.stats as ss

# TODO: smooth the cortical and cerebellar maps before getting the overlap/effects

wkdir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'

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
    data_load, data_recall = calc_contrast(data, info)
    return data_load,data_recall

def plot_overlap_cerebellum(subject = "group", 
                            phase = 0, 
                            scale = None, 
                            threshold = None):
    """
    Makes an overlap plot for the cerebellum 
    """
    load_eff,dir_eff=load_contrast(ses_id = 'ses-02',subj = subject,atlas_space='SUIT3',phase=phase)
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
                        scale = None, 
                        threshold = None):
    """
    Makes an overlap plot for the cerebellum 
    """
    load_eff,dir_eff= load_contrast(ses_id = 'ses-02',subj = "group",atlas_space='fs32k',phase=phase)
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

def calc_load_recall_effect(type = 'CondAll', 
                            threshold = 0, 
                            binarize = False):
    """
    code to create contrasts for load and recall.
    can be used to evaluate overlap between the contrasts during encoding and retrieval

    Args:
        type (str) - type of data you want to use. defaults to 'CondAll'
        threshold (float) - percentile value you want to use for thresholding. 0 means no threshold
        binarize (bool) - boolean variable used when you want to threshold the data
    Returns:
        effect_obj (cift2image) cifti object containing effects of load and recall in encoding and retrieval
    """
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")
    tensor = []
    tensor_cerebellum, info_tsv, _ = ds.get_dataset(gl.base_dir,"WMFS",atlas="SUIT3",sess="ses-02",type=type, info_only=False)
    tensor_cortex, info_tsv, _ = ds.get_dataset(gl.base_dir,"WMFS",atlas="fs32k",sess="ses-02",type=type, info_only=False)
    tensor.append(tensor_cerebellum)
    tensor.append(tensor_cortex)

    # get cortical and cerebellar atlases into a list
    atlases = []
    atlas_cerebellum, a_info = am.get_atlas('SUIT3',gl.atlas_dir)
    atlas_cortex, a_info = am.get_atlas('fs32k',gl.atlas_dir)
    atlases.append(atlas_cerebellum)
    atlases.append(atlas_cortex)

    # get list of subjects
    T = Data.get_participants()
    subject_list = list(T.participant_id.values)
    # NOTE: group is the subject that is created by averaging the data across all subjects 
    subject_list.append("group")

    # add the group subject
    # NOTE: group subject is the subject created by averaging over all subjcts data
    cifti_file = nb.load(Data.data_dir.format("group") + f"/group_space-{structure}_ses-02_{type}.dscalar.nii")
    cifti_data = cifti_file.get_fdata()
    # include the subject labeled group as well
    tensor[ss] = np.append(tensor[ss], cifti_data[np.newaxis, ...], axis = 0)

    for s, subject in enumerate(subject_list):     
        effect_obj = []     
        for ss, structure in enumerate(['SUIT3', 'fs32k']):
            

            effect_list = []
            col_names = []
            for p, phase in enumerate(['Enc', 'Ret', 'overall']):

                print(f"- Doing {phase} {structure} {subject}")
                
                # get data for current subject
                data = tensor[ss][s, :, :]

                # get data for the current phase
                if phase == "overall":
                    # averaging over encoding and retrieval
                    idx = True*np.ones([len(info_tsv,)], dtype=bool)
                else:
                    idx = info_tsv.phase == p
                    
                info = info_tsv.loc[idx]
                data_phase = data[idx, :]
                
                # get load and recall contrasts for the current subject
                data_effect = calc_contrast(data_phase, info)
                data_load = data_effect[0, :]
                data_recall = data_effect[1, :]

                if threshold != 0:
                    # threshold the data
                    data_load =ra.threshold_map(data_load, threshold, binarize = binarize)
                    data_recall =ra.threshold_map(data_recall, threshold, binarize = binarize)

                effect_list.append(data_load.reshape(-1, 1).T)
                col_names.append(f"{phase}_load")

                effect_list.append(data_recall.reshape(-1, 1).T)
                col_names.append(f"{phase}_recall")

            data_effect = np.concatenate(effect_list, axis = 0)
            # save ciftis as dscalar
            effect_obj.append(atlases[ss].data_to_cifti(data_effect, row_axis = col_names))

            # save the cifti file
            save_path = f"{wkdir}/data/{subject}"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            nb.save(effect_obj[ss], f"{save_path}/load_recall_space_{structure}_{type}_{subject}.dscalar.nii")
    return 

def calc_reliability_effect(threshold = 0, binarize = False):
    """Calculates reliability of the contrast across halves
    """

    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")
    tensor = []
    tensor_cerebellum, info_tsv, _ = ds.get_dataset(gl.base_dir,"WMFS",atlas="SUIT3",sess="ses-02",type="CondHalf", info_only=False)
    tensor_cortex, info_tsv, _ = ds.get_dataset(gl.base_dir,"WMFS",atlas="fs32k",sess="ses-02",type="CondHalf", info_only=False)
    tensor.append(tensor_cerebellum)
    tensor.append(tensor_cortex)

    # get cortical and cerebellar atlases into a list
    atlases = []
    atlas_cerebellum, a_info = am.get_atlas('SUIT3',gl.atlas_dir)
    atlas_cortex, a_info = am.get_atlas('fs32k',gl.atlas_dir)
    atlases.append(atlas_cerebellum)
    atlases.append(atlas_cortex)

    # get list of subjects
    T = Data.get_participants()
    subject_list = list(T.participant_id.values)
    # NOTE: group is the subject that is created by averaging the data across all subjects 
    subject_list.append("group")
    DD = []
    for s, subject in enumerate(subject_list):     
        for ss, structure in enumerate(['SUIT3', 'fs32k']):
            # add the group subject
            # NOTE: group subject is the subject created by averaging over all subjcts data
            cifti_file = nb.load(Data.data_dir.format("group") + f"/group_space-{structure}_ses-02_CondHalf.dscalar.nii")
            cifti_data = cifti_file.get_fdata()
            # include the subject labeled group as well
            tensor[ss] = np.append(tensor[ss], cifti_data[np.newaxis, ...], axis = 0)

            # R_vox_list = []
            for p, phase in enumerate(['Enc', 'Ret', 'overall']):

                print(f"- Doing {phase} {structure} {subject}")
                
                # get data for current subject
                data = tensor[ss][s, :, :]

                # get data for the current phase
                if phase == "overall":
                    # averaging over encoding and retrieval
                    idx = True*np.ones([len(info_tsv,)], dtype=bool)
                else:
                    idx = info_tsv.phase == p

                # get data for each half
                load_half_list = []
                recall_half_list = []
                effect_half_list = []
                for half in [1, 2]:
                    idx_half = info_tsv.half == half
                    info_half = info_tsv.loc[idx_half & idx]
                    data_half = data[idx_half & idx, :]
                
                    # get load and recall contrasts for the current subject
                    effect_half = get_contrast(data_half, info_half)
                    load_half = effect_half[0]
                    recall_half = effect_half[1]

                    load_half_list.append(load_half.reshape(-1, 1))
                    recall_half_list.append(recall_half.reshape(-1, 1))

                R_load = calculate_R(load_half_list[0].T, load_half_list[1].T)
                R_recall = calculate_R(recall_half_list[0].T, recall_half_list[1].T)


                R_dict = {}
                R_dict["dataset"] = np.repeat("WMFS", 2, axis=0)
                R_dict["ses_id"] =np.repeat("ses-02", 2, axis=0)
                R_dict["phase"] = np.repeat(phase, 2, axis=0)
                R_dict["structure"] = np.repeat(structure, 2, axis=0)
                R_dict["R"] = np.array([R_load,R_recall]) 
                R_dict["effect"] = np.array(["load","recall"])
                R_dict["sn"] = np.repeat(subject, 2, axis=0)

                R_df = pd.DataFrame(R_dict, index = [0, 1])
                DD.append(R_df)

    return pd.concat(DD)

def overlap_corr(type = 'CondAll'):
    """
    quantifying the overlap between the two contrasts by calculating the overlap between maps
    """

    # get Dataset class for your dataset
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")

    # get list of subjects
    T = Data.get_participants()
    subject_list = list(T.participant_id.values)
    subject_list.append("group")
    D = []
    for s, subject in enumerate(subject_list):     
        for ss, structure in enumerate(['SUIT3', 'fs32k']):
            # load the cifti object for the subject
            fpath = f"{wkdir}/data/{subject}"
            cifti_obj = nb.load(f"{fpath}/load_recall_space_{structure}_{type}_{subject}.dscalar.nii")
            data_cifti = cifti_obj.get_fdata()

            # get the names of the contrasts
            con_names = list(cifti_obj.header.get_axis(0).name)

            # loop over phases
            for p, phase in enumerate(['Enc', 'Ret', 'overall']):
                # get the indices of contrasts in the current phase
                effect_idx = [con_names.index(con) for con in con_names if phase in con]

                # get the rows corresponding to the two contrasts
                data_effect = data_cifti[effect_idx, :]

                # calculate correlations
                corr= calculate_R(data_effect[0, :], data_effect[1, :])
                R_dict = {}
                R_dict["dataset"] = "WMFS"
                R_dict["ses_id"] ="ses-02"
                R_dict["phase"] = phase
                R_dict["structure"] = structure
                R_dict["R"] = corr
                R_dict["sn"] = subject
                R_dict["sn_id"] = s

                R_df = pd.DataFrame(R_dict, index = [0])
                D.append(R_df)
            
    return pd.concat(D)

def conjunction_map_cortex(type = "CondAll", phase = "Enc"):

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

        # load the cifti object for the subject
        fpath = f"{wkdir}/data/{subject}"
        cifti_obj = nb.load(f"{fpath}/load_recall_space_fs32k_{type}_{subject}.dscalar.nii")
        data_cifti = cifti_obj.get_fdata()

        # get the names of the contrasts
        con_names = cifti_obj.header.get_axis(0).name

        # get the indices for the current phase
        idx_phase = np.array([phase in name for name in con_names])

        # get the indices for the load effect
        idx_load = np.array(["load" in name for name in con_names])

        # the indices for the recall effect
        idx_recall = np.array(["recall" in name for name in con_names])

        # get the data for load
        data_load = data_cifti[idx_phase & idx_load, :]
        data_load_subs.append(data_load)

        # get the data for recall
        data_recall = data_cifti[idx_phase & idx_recall, :]
        data_recall_subs.append(data_recall)

    # create array containing data for all subjects
    data_load_arr = np.concatenate(data_load_subs, axis = 0)
    data_recall_arr = np.concatenate(data_recall_subs, axis = 0)

    # do the test per hemisphere
    load_sub_list = nt.surf_from_cifti(atlas.data_to_cifti(data_load_arr))
    recall_sub_list = nt.surf_from_cifti(atlas.data_to_cifti(data_recall_arr))
    
    z_val_load = []
    z_val_recall = []
    for h, hemi in enumerate(["L", "R"]):

        t_val_load, p_val_load = ttest_1samp(load_sub_list[h], 0)
        z_val_load.append(norm.isf(p_val_load))

        t_val_recall, p_val_recall = ttest_1samp(recall_sub_list[h], 0)
        z_val_recall.append(norm.isf(p_val_recall))

    return z_val_load, z_val_recall

def conjunction_map_cerebellum(type = "CondAll", phase = "Enc"):

    """
    """
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")
    # get list of subjects
    T = Data.get_participants()
    subject_list = list(T.participant_id.values)

    # get the atlas object for cifit/nifti creation
    atlas, ainfo = am.get_atlas("SUIT3", Data.atlas_dir)

    # loop over subjects and load the cifti
    data_load_subs = []
    data_recall_subs = []
    for s, subject in enumerate(subject_list):

        # load the cifti object for the subject
        fpath = f"{wkdir}/data/{subject}"
        cifti_obj = nb.load(f"{fpath}/load_recall_space_SUIT3_{type}_{subject}.dscalar.nii")
        data_cifti = cifti_obj.get_fdata()

        # get the names of the contrasts
        con_names = cifti_obj.header.get_axis(0).name

        # get the indices for the current phase
        idx_phase = np.array([phase in name for name in con_names])

        # get the indices for the load effect
        idx_load = np.array(["load" in name for name in con_names])

        # the indices for the recall effect
        idx_recall = np.array(["recall" in name for name in con_names])

        # get the data for load
        data_load = data_cifti[idx_phase & idx_load, :]
        data_load_subs.append(data_load)

        # get the data for recall
        data_recall = data_cifti[idx_phase & idx_recall, :]
        data_recall_subs.append(data_recall)

    # create array containing data for all subjects
    data_load_arr = np.concatenate(data_load_subs, axis = 0)
    data_recall_arr = np.concatenate(data_recall_subs, axis = 0)

    # do test
    t_val_load, p_val_load = ttest_1samp(data_load_arr, 0)
    z_val_load = norm.isf(p_val_load)

    t_val_recall, p_val_recall = ttest_1samp(data_recall_arr, 0)
    z_val_recall = norm.isf(p_val_recall)

    # convert to nifti

    return z_val_load, z_val_recall

def plot_contrast_cerebellum(subject, phase = "Enc", effect = "load", save = False):
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
    if save:
        ax.write_image(f"{wkdir}/figures/flatmap_{phase}_{effect}_{subject}.svg")
    return 

def plot_contrast_cortex(subject, phase, effect, save = False):
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
    if save:
        ax.write_image(f"{wkdir}/figures/cortex_left_{phase}_{effect}_{subject}.svg")
    return ax


if __name__=="__main__":
    plot_overlap_cortex()
