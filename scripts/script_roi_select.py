#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to make label files for the cortical and cerebellar rois

Created on 12/09/2022 at 5:25 PM
Author: Ladan Shahshahani
"""
# import packages
import sys
sys.path.append('../Functional_Fusion') 
sys.path.append('../cortico-cereb_connectivity') 
sys.path.append('..')

import numpy as np
import pandas as pd
from pathlib import Path

# modules from functional fusion
from atlas_map import *
from dataset import *

#
import os
import nibabel as nb
import nitools as nt


# def make_roi_cerebellum(cifti_img, info, threshold, atlas_space = "SUIT3", contrast = "Verbal2Back"):
#     """
#     creates label nifti for roi cerebellum
#     Args:
#     cifti_img (nb.Cifti2) - cifti image of the extracted data
#     info (pd.DataFrame) - pandas dataframe representing info for the dataset
#     threshold (int) - percentile value used in thresholding
#     contrast (str) - name of the contrast
#     Returns:
#     nifti_img (nb.nifti) - nifti of the label created
#     """
#     # get suit data
#     data = cifti_img.get_fdata()
#     # get the row index corresponding to the contrast
#     info_con = info.loc[info.cond_name == contrast]

#     if info_con.empty:
#         raise ValueError("No rows found for contrast '{}'".format(contrast))
#     # get the map for the contrast of interest
#     con_map  = data[info_con.cond_num.values -1, :]

#     # get threshold value (ignoring nans)
#     percentile_value = np.nanpercentile(con_map, q=threshold)

#     # apply threshold
#     thresh_data = con_map > percentile_value
#     # convert 0 to nan
#     thresh_data[thresh_data != False] = np.nan

#     # create an instance of the atlas (will be used to convert data to nifti)
#     atlas, a_info = am.get_atlas(atlas_space,atlas_dir)
#     nifti_img = atlas.data_to_nifti(1*thresh_data)
#     return nifti_img

def make_roi_cerebellum(cifti_img, info, threshold, atlas_space = "SUIT3", contrast1 = "x",contrast2='y'):
    """
    creates label nifti for roi cerebellum
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    contrast1 (str) - name of the first contrast
    contrast2 (str) - name of the second contrast
    Returns:
    nifti_img (nb.nifti) - nifti of the label created
    """
    # get suit data
    data = cifti_img.get_fdata()
    # get the row index corresponding to the contrast 1
    info_con1 = info.loc[info.names == contrast1]
    print(info_con1)

    if info_con1.empty:
        raise ValueError("No rows found for contrast '{}'".format(contrast1))
        
    # get the map for the contrast 1 of interest
    con_map1  = data[info_con1.cond_num_uni.values -1, :]
    print(con_map1)
    # get the row index corresponding to the contrast 2
    info_con2 = info.loc[info.names == contrast2]
    print(info_con2)

    if info_con2.empty:
        raise ValueError("No rows found for contrast '{}'".format(contrast2))
        
    # get the map for the contrast 2 of interest
    con_map2  = data[info_con2.cond_num_uni.values -1, :]
    print(con_map2)
    
    # calculate the difference between the two contrasts
    con_diff = con_map1 - con_map2

    print(con_diff)

    # get threshold value (ignoring nans)
    percentile_value = np.nanpercentile(con_diff, q=threshold)

    # apply threshold
    thresh_data = con_diff > percentile_value
    # convert 0 to nan
    thresh_data[thresh_data != False] = np.nan

    # create an instance of the atlas (will be used to convert data to nifti)
    atlas, a_info = am.get_atlas(atlas_space,atlas_dir)
    nifti_img = atlas.data_to_nifti(1*thresh_data)
    return nifti_img



# def make_roi_cortex(cifti_img, info, threshold, contrast = "Verbal2Back"):
#     """
#     creates label giftis for left and right hemisphere of the cortex
#     Args:
#     cifti_img (nb.Cifti2) - cifti image of the extracted data
#     info (pd.DataFrame) - pandas dataframe representing info for the dataset
#     threshold (int) - percentile value used in thresholding
#     contrast (str) - name of the contrast
#     Returns:
#     gifti_img (list) - list of label giftis for left and right hemispheres
#     """
#     # get data for left and right hemisphere
#     data_list = nt.surf_from_cifti(cifti_img)

#     # threshold and create label
#     gifti_img = []
#     for i, name in zip([0, 1], ['CortexLeft', 'CortexRight']):
#         # get data for the hemisphere
#         data = data_list[i]

#         # get the contrast map
#         # get the row index corresponding to the contrast
#         info_con = info.loc[info.cond_name == contrast]
#         # get the map for the contrast of interest
#         con_map  = data[info_con.cond_num.values -1, :]
#         print(con_map)
#         # get threshold value (ignoring nans)
#         percentile_value = np.nanpercentile(con_map, q=threshold)

#         # apply threshold
#         thresh_data = con_map > percentile_value
#         # convert 0 to nan
#         thresh_data[thresh_data != False] = np.nan
#         # create label gifti
#         gifti_img.append(nt.make_label_gifti(1*thresh_data.T, anatomical_struct=name))
#     return gifti_img

def make_roi_cortex(cifti_img, info, threshold, contrast1="x", contrast2="y"):
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

        # get the contrast map for contrast 1
        # get the row index corresponding to contrast 1
        info_con1 = info.loc[info.names == contrast1]
        # get the map for contrast 1 of interest
        con_map1 = data[info_con1.cond_num_uni.values -1, :]

        # get the contrast map for contrast 2
        # get the row index corresponding to contrast 2
        info_con2 = info.loc[info.names == contrast2]
        # get the map for contrast 2 of interest
        con_map2 = data[info_con2.cond_num_uni.values -1, :]

        # get the difference between the two contrast maps
        contrast_diff = con_map1 - con_map2

        # get threshold value (ignoring nans)
        percentile_value = np.nanpercentile(contrast_diff, q=threshold)

        # apply threshold
        thresh_data = contrast_diff > percentile_value
        # convert 0 to nan
        thresh_data[thresh_data != False] = np.nan
        # create label gifti
        gifti_img.append(nt.make_label_gifti(1*thresh_data.T, anatomical_struct=name))
    return gifti_img


def make_roi_label(dataset_name = "MDTB", 
                   contrast1 = "VerbGen", 
                   contrast2 = 'Y',
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
    Data = get_dataset_class(base_dir, dataset=dataset_name)

    # load data 
    cifti_cerebellum = nb.load(Data.data_dir.format("group") + f"/group_space-SUIT3_{ses_id}_Condhalf.dscalar.nii")
    cifti_cortex = nb.load(Data.data_dir.format("group") + f"/group_space-fs32k_{ses_id}_Condhalf.dscalar.nii")
    
    # load info (will be used to select contrast)
    info_tsv = pd.read_csv(Data.data_dir.format("group") + f"/group_ses-archi_info-Condhalf.tsv", sep="\t")

    # label files for the cerebellum and cortex
    roi_nifti = make_roi_cerebellum(cifti_cerebellum, info_tsv, threshold, atlas_space = "SUIT3", contrast1 = "speech-half1", contrast2='non_speech-half1')
    roi_gifti = make_roi_cortex(cifti_cortex, info_tsv, threshold, contrast1 = "speech-half1",contrast2 = 'non_speech-half1')

    # save the nifti image
    nb.save(roi_nifti, Data.atlas_dir + '/tpl-SUIT' + f'/atl-speechvsnonspeech_space-SUIT_dseg.nii')
    
    # save label gifti images
    for i, h in enumerate(['L', 'R']):
        nb.save(roi_gifti[i], Data.atlas_dir + '/tpl-fs32k' + f'/speechvsnonspeech.32k.{h}.label.gii')

    return




if __name__ == "__main__":

    base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
    conn_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/connectivity'
    if not Path(base_dir).exists():
        base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
        conn_dir = '/cifs/diedrichsen/data/Cerebellum/connectivity'
    atlas_dir = base_dir + '/Atlases'




    make_roi_label(dataset_name = "IBC", 
                   contrast1 = "speech-half1", 
                   contrast2 = 'non_speech-half1',
                   ses_id = "ses-archi", 
                   threshold = 80)