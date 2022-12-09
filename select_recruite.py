#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions of atlas definition and atlas mapping

Created on 12/09/2022 at 5:25 PM
Author: Ladan Shahshahani
"""
# import packages
import sys
sys.path.append('../Functional_Fusion') 
sys.path.append('../cortico-cereb_connectivity') 

import numpy as np
import pandas as pd
from pathlib import Path

from atlas_map import *
from dataset import *

import os
import nibabel as nb



# WHAT TO DO?
# create an instance of the dataset* 

# extract the data within atlas for cerebellum*

# extract the data within atlas for cortex*

# extract data within selected parcellation for cerebellum*

# extract data within selected parcellation for cortex*

# use connectivity model

# regress cerebellar data onto cortical data 
    # get residuals

# get summary (a wrapper)

# set the directory of your dataset here:
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'

data_dir = base_dir + '/WMFS'
atlas_dir = base_dir + '/Atlases'

# 1. run this case if you have not extracted data for the atlas
def extract_suit(dataSet, ses_id, type, atlas):
    # create an instance of the dataset class
    dataset = dataSet(data_dir)

    # extract data for suit atlas
    dataset.extract_all_suit(ses_id,type,atlas)

    return

# 2. run this case if you have not extracted data for the atlas    
def extract_fs32K(dataSet, ses_id, type):
    # create an instance of the dataset class
    dataset = dataSet(data_dir)

    # extract data for suit atlas
    dataset.extract_all_fs32k(ses_id,type)
    return

# 3. run after you have extracted data for the cerebellum
def agg_data_cerebellum(data_dir, subj_id, ses_id = 2, atlas = 'SUIT3', parcel = 'MDTB10', type = 'CondHalf'):

    """
    Args:
    """
    df = pd.DataFrame()
    # get the data for thee current participant
    data_file = nb.cifti2.load(os.path.join(data_dir, 'derivatives', subj_id, 'data', f'{subj_id}_space-{atlas}_ses-{ses_id:02d}_{type}.dscalar.nii'))
    data = data_file.get_fdata()
    # get the corresponding tsv file for condition information
    info_tsv = pd.read_csv(os.path.join(data_dir, 'derivatives', subj_id, 'data', f'{subj_id}_ses-{ses_id:02d}_info-{type}.tsv'), sep = '\t')
    
    # get the resolution for suit
    suit_res = atlas[-1]

    # get the gray matter mask for the atlas reslution
    mask_img = os.path.join(atlas_dir, 'tpl-SUIT', f'tpl-SUIT_res-{suit_res}_gmcmask.nii')

    # get the label file for the parcellation
    label_img = os.path.join(atlas_dir, 'tpl-SUIT', f'atl-{parcel}_space-SUIT_dseg.nii')

    # create an instance of atlas parcel
    Parcel = AtlasVolumeParcel('cerebellum',label_img,mask_img)

    # get label names (will be used to create dataframe for scatterplot)
    parcel_axis = Parcel.get_parcel_axis()
    parcel_names = parcel_axis.name

    # aggregate data over the parcel
    array_parcel = Parcel.agg_data(data)

    # creating the dataframe
    for region in np.arange(array_parcel.shape[1]):
        dd = info_tsv.copy()
        # get region values
        array_region = array_parcel[:, region]

        # set region name
        dd['values'] = array_region

        # set region number
        dd['region_number'] = region + 1

        # set the region name
        dd['region_name'] = parcel_names[region]

        # set the subject id
        dd['subj_id'] = subj_id

        df = pd.concat([df, dd], ignore_index = True)


    return df

# 4. run after you have extracted data for the cortex 
def agg_data_cortex(data_dir, subj_id, ses_id = 2, atlas = 'fs32k', parcel = 'ROI', type = 'CondHalf'):
    # doing both hemispheres
    hemi = ['L', 'R']
    # other options for parcel: Icosahedron-42_Sym
    df = pd.DataFrame()
    # get the data for thee current participant
    data_file = nb.cifti2.load(os.path.join(data_dir, 'derivatives', subj_id, 'data', f'{subj_id}_space-{atlas}_ses-{ses_id:02d}_{type}.dscalar.nii'))
    data = data_file.get_fdata()
    # get the corresponding tsv file for condition information
    info_tsv = pd.read_csv(os.path.join(data_dir, 'derivatives', subj_id, 'data', f'{subj_id}_ses-{ses_id:02d}_info-{type}.tsv'), sep = '\t')

    # get brain models
    bmf = data_file.header.get_axis(1)
    data_list = []
    parcel_list = []
    for idx, (nam,slc,bm) in enumerate(bmf.iter_structures()):
        # get the data corresponding to the brain structure
        data_hemi = data[:, slc]

        # get name to be passed on to the AtlasSurfaceParcel object
        name = nam[16:].lower()

        # create atlas parcel object
        # label_img = os.path.join(atlas_dir, 'tpl-fs32k', f'Icosahedron-642_Sym.32k.{hemi[idx]}.label.gii')
        label_img = os.path.join(atlas_dir, 'tpl-fs32k', f'{parcel}.32k.{hemi[idx]}.label.gii')
        mask_img = os.path.join(atlas_dir, 'tpl-fs32k', f'tpl-fs32k_hemi-{hemi[idx]}_mask.label.gii')
        Parcel = AtlasSurfaceParcel(name,label_img,mask_img)
        data_list.append(Parcel.agg_data(data_hemi))

        # get label names (will be used to create dataframe for scatterplot)
        parcel_axis = Parcel.get_parcel_axis()
        parcel_list.append(parcel_axis.name)

    parcel_names = np.concatenate(parcel_list, axis = 0)

    # concatenate into a single array
    array_parcel = np.concatenate(data_list, axis = 1)

    # creating the dataframe
    for region in np.arange(array_parcel.shape[1]):
        dd = info_tsv.copy()
        # get region values
        array_region = array_parcel[:, region]

        # set region name
        dd['values'] = array_region

        # set region number
        dd['region_number'] = region + 1

        # set the region name
        dd['region_name'] = parcel_names[region]

        # set the subject id
        dd['subj_id'] = subj_id

        df = pd.concat([df, dd], ignore_index = True)

    return df

# 4. Alternative: use connectivity model to predict cerebellar activation
def predict_cerebellum():
    return

# 5. regress cerebellar data onto step 4 (or 4:alternative)
def regress():
    return

if __name__ == "__main__":
    a = agg_data_cerebellum(data_dir, "sub-01", ses_id = 2, atlas = 'SUIT3', parcel = 'MDTB10', type = 'CondHalf')
