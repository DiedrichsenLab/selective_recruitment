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


def make_roi_cerebellum(cifti_img, info, threshold, atlas_space = "SUIT3", contrast = "Verbal2Back"):
    """
    creates label nifti for roi cerebellum
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    contrast (str) - name of the contrast
    Returns:
    nifti_img (nb.nifti) - nifti of the label created
    """
    # get suit data
    data = cifti_img.get_fdata()
    # get the row index corresponding to the contrast
    info_con = info.loc[info.cond_name == contrast]
    # get the map for the contrast of interest
    con_map  = data[info_con.cond_num.values -1, :]

    # get threshold value (ignoring nans)
    percentile_value = np.nanpercentile(con_map, q=threshold)

    # apply threshold
    thresh_data = con_map > percentile_value
    # convert 0 to nan
    thresh_data[thresh_data != False] = np.nan

    # create an instance of the atlas (will be used to convert data to nifti)
    atlas, a_info = am.get_atlas(atlas_space,atlas_dir)
    nifti_img = atlas.data_to_nifti(1*thresh_data)
    return nifti_img


def make_roi_cortex(cifti_img, info, threshold, contrast = "Verbal2Back"):
    """
    creates label giftis for left and right hemisphere of the cortex
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    contrast (str) - name of the contrast
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

        # get the contrast map
        # get the row index corresponding to the contrast
        info_con = info.loc[info.cond_name == contrast]
        # get the map for the contrast of interest
        con_map  = data[info_con.cond_num.values -1, :]

        # get threshold value (ignoring nans)
        percentile_value = np.nanpercentile(con_map, q=threshold)

        # apply threshold
        thresh_data = con_map > percentile_value
        # convert 0 to nan
        thresh_data[thresh_data != False] = np.nan
        # create label gifti
        gifti_img.append(nt.make_label_gifti(1*thresh_data.T, anatomical_struct=name))
    return gifti_img


def make_roi_label(dataset_name = "MDTB", 
                   contrast = "Verbal2Back", 
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
    cifti_cerebellum = nb.load(Data.data_dir.format("group") + f"/group_space-SUIT3_{ses_id}_CondAll.dscalar.nii")
    cifti_cortex = nb.load(Data.data_dir.format("group") + f"/group_space-fs32k_{ses_id}_CondAll.dscalar.nii")
    
    # load info (will be used to select contrast)
    info_tsv = pd.read_csv(Data.data_dir.format("group") + f"/group_ses-s1_info-CondAll.tsv", sep="\t")

    # label files for the cerebellum and cortex
    roi_nifti = make_roi_cerebellum(cifti_cerebellum, info_tsv, threshold, atlas_space = "SUIT3", contrast = "Verbal2Back")
    roi_gifti = make_roi_cortex(cifti_cortex, info_tsv, threshold, contrast = "Verbal2Back")

    # save the nifti image
    nb.save(roi_nifti, Data.atlas_dir + '/tpl-SUIT' + f'/atl-{contrast}_space-SUIT_dseg.nii')
    
    # save label gifti images
    for i, h in enumerate(['L', 'R']):
        nb.save(roi_gifti[i], Data.atlas_dir + '/tpl-fs32k' + f'/{contrast}.32k.{h}.label.gii')

    return