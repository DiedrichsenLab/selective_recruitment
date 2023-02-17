#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to make label files for the cortical and cerebellar rois

Created on 12/09/2022 at 5:25 PM
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

#
import os
import nibabel as nb
import nitools as nt


def make_roi_cerebellum(cifti_img, info, threshold, atlas_space = "SUIT3", condition_1 = "x",condition_2='y'):
    """
    creates label nifti for roi cerebellum
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    condition_1 (str) - name of the first condition
    condition_2 (str) - name of the second condition (optional for creating a contrast between the conditions )
    Returns:
    nifti_img (nb.nifti) - nifti of the label created
    """
    # get suit data
    data = cifti_img.get_fdata()

    # get the row index corresponding to the contrast 1
    info_con_1 = info.loc[info.names == condition_1]
    if info_con_1.empty:
        raise ValueError("No rows found for contrast '{}'".format(condition_1))
        
    # get the map for the contrast 1 of interest
    con_map_1  = data[info_con_1.cond_num_uni.values -1, :]


    if condition_2 != 'y':
        # get the row index corresponding to the contrast 2
        info_con_2 = info.loc[info.names == condition_2]
        if info_con_2.empty:
            raise ValueError("No rows found for contrast '{}'".format(condition_2))
        
        # get the map for the contrast 2 of interest
        con_map_2  = data[info_con_2.cond_num_uni.values -1, :]

        # calculate the difference between the two contrasts
        con_final = con_map_1 - con_map_2
    else:

        con_final = con_map_1

    # get threshold value (ignoring nans)
    percentile_value = np.nanpercentile(con_final, q=threshold)

    # apply threshold
    thresh_data = con_final > percentile_value
    # convert 0 to nan
    thresh_data[thresh_data != False] = np.nan

    # create an instance of the atlas (will be used to convert data to nifti)
    atlas, a_info = am.get_atlas(atlas_space,atlas_dir)
    nifti_img = atlas.data_to_nifti(1*thresh_data)
    return nifti_img

def make_roi_cortex(cifti_img, info, threshold, condition_1="x", condition_2="y"):
    """
    creates label giftis for left and right hemisphere of the cortex
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    contrast1 (str) - name of the first contrast
    contrast2 (str) - name of the second contrast
    Returns:
    gifti_img (list) - list of label giftis for left and right hemispheres
    """
    # get data for left and right hemisphere
    data_list = nt.surf_from_cifti(cifti_img)

    # threshold and create label
    gifti_img = []
    for i, name in zip([0, 1], ['CortexLeft', 'CortexRight']):
        # get data for the hemisphere
        data = data_list[i]

        # get the row index corresponding to contrast 1
        info_con_1 = info.loc[info.names == condition_1]
        # get the map for contrast 1 of interest
        con_map_1 = data[info_con_1.cond_num_uni.values -1, :]

        if condition_2 != 'y':
            # get the row index corresponding to contrast 2
            info_con_2 = info.loc[info.names == condition_2]
            # get the map for contrast 2 of interest
            con_map2 = data[info_con_2.cond_num_uni.values -1, :]

            # get the difference between the two contrast maps
            con_final = con_map_1 - con_map2
        else:
            con_final = con_map_1

        # get threshold value (ignoring nans)
        percentile_value = np.nanpercentile(con_final, q=threshold)

        # apply threshold
        thresh_data = con_final > percentile_value
        # convert 0 to nan
        thresh_data[thresh_data != False] = np.nan
        # create label gifti
        gifti_img.append(nt.make_label_gifti(1*thresh_data.T, anatomical_struct=name))
    return gifti_img


def make_roi_label(dataset_name = "MDTB", 
                   condition_1 = "x", 
                   condition_2 = 'y',
                   ses_id = "ses-s1", 
                   threshold = 80):
    """
    Uses a group contrast to create label files (nifti and gifti)
    
    Args:
        dataset_name (str) - dataset
        contrast (str) - name of the contrast (see info tsv file) to be used for roi creation
        ses_id (str) - session id where the contrast of interest is
        threshold (int) - percentile value (1-100)

    """
    # get Dataset class for your dataset
    Data = ds.get_dataset_class(gl.base_dir, dataset=dataset_name)

    # load data 
    cifti_cerebellum = nb.load(Data.data_dir.format("group") + f"/group_space-SUIT3_{ses_id}_Condhalf.dscalar.nii")
    cifti_cortex = nb.load(Data.data_dir.format("group") + f"/group_space-fs32k_{ses_id}_Condhalf.dscalar.nii")
    
    # load info (will be used to select contrast)
    info_tsv = pd.read_csv(Data.data_dir.format("group") + f"/group_ses-archi_info-Condhalf.tsv", sep="\t")

    # label files for the cerebellum and cortex
    roi_nifti = ra.make_roi_cerebellum(cifti_cerebellum, info_tsv, threshold, atlas_space = "SUIT3", contrast = "Verbal2Back")
    roi_gifti = ra.make_roi_cortex(cifti_cortex, info_tsv, threshold, contrast = "Verbal2Back")

    if condition_2 != 'y':
        # save the nifti image
        nb.save(roi_nifti, save_dir + '/tpl-SUIT' + f'/atl-{condition_1}_vs_{condition_2}_space-SUIT_dseg.nii')
        
        # save label gifti images
        for i, h in enumerate(['L', 'R']):
            nb.save(roi_gifti[i], save_dir + '/tpl-fs32k' + f'/vertical-{condition_1}_vs_{condition_2}.32k.{h}.label.gii')

    else: 
        # save the nifti image
        nb.save(roi_nifti, save_dir + '/tpl-SUIT' + f'/atl-{condition_1}_space-SUIT_dseg.nii')
        
        # save label gifti images
        for i, h in enumerate(['L', 'R']):
            nb.save(roi_gifti[i], save_dir + '/tpl-fs32k' + f'/{condition_1}.32k.{h}.label.gii')

    return




if __name__ == "__main__":

    base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
    conn_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/connectivity'
    if not Path(base_dir).exists():
        base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
        conn_dir = '/cifs/diedrichsen/data/Cerebellum/connectivity'
        save_dir = '/cifs/diedrichsen/data/Cerebellum/Language/atlases'
    if not Path(base_dir).exists():
        base_dir = '/srv/diedrichsen/data/FunctionalFusion'
        conn_dir = '/srv/diedrichsen/data/Cerebellum/connectivity'
        save_dir = '/srv/diedrichsen/data/Cerebellum/Language/atlases'
    atlas_dir = base_dir + '/Atlases'
    


    make_roi_label(dataset_name = "IBC", 
                   condition_1 = "vertical_checkerboard-half1",    
                   ses_id = "ses-archi", 
                   threshold = 80)