#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to make label files for the cortical and cerebellar rois

Created on 07/03/2023
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
import SUITPy.flatmap as flatmap


def get_contrast(data, info):
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
    data_load = data[idx_effect_load, :]/np.sum(idx_effect_load) - data[idx_base_load, :]/np.sum(idx_base_load)
    
    # calculate the effect of recall direction
    ## get the index for baseline condition
    idx_base_recall = info.recall == 0
    ## get the index for effect condition
    idx_effect_recall = info.recall == 1
    ## recall effect
    data_recall = data[idx_effect_recall, :]/np.sum(idx_effect_recall) - data[idx_base_recall]/np.sum(idx_base_recall)
    
    return data_load, data_recall

def plot_overlap_cerebellum(dataset = 'WMFS', 
                            type = 'CondAll',
                            ses_id = 'ses_02',
                            subject = "group",
                            atlas = 'SUIT3', 
                            threshold = 80, 
                            binarize = True,
                            ):
    """get a map of overlap/unique activated areas for certain contrasts.
    1. loads in the cifti files in the cerebellar and cortical atlas spaces
    2. Calculates the desired contrast (uses the info file to get conditions)
    3. applies a threshold to the desired contrast using the percentile value provided
    4. (optional?) binarizes the thresholded maps and assigns integer values to the 
    voxels/vertices passing the threshold
    5. gets the overlap between the two contrasts
    6. (if binarized) creates a label file with labels assigned to unique and overlap regions
    """
    # create a dataset class for the dataset
    # get Dataset class for your dataset
    Data = ds.get_dataset_class(gl.base_dir, dataset=dataset)
    
    # load data 
    cifti_file = nb.load(Data.data_dir.format(subject) + f"/group_space-{atlas}_{ses_id}_{type}.dscalar.nii")
    cifti_data = cifti_file.get_fdata()
    # load info (will be used to select and calculate contrasts)
    info_tsv = pd.read_csv(Data.data_dir.format(subject) + f"/group_{ses_id}_info-{type}.tsv", sep="\t")

    # loop over phases
    for p, phase in enumerate(["Enc", "Ret", "overall"]):
        # get the index for the phase
        if phase == "overall":
            idx = True*np.ones([len(info_tsv,)], dtype=bool)
        else:
            idx = info_tsv.phase == p
            
        info = info_tsv.loc[idx]
        data = cifti_data[idx, :]
        
        # get contrasts
        data_load, data_recall = get_contrast(data, info)
        
        # threshold the data
        thresh_load =ra.threshold_map(data_load, threshold, binarize = binarize)
        thresh_recall =ra.threshold_map(data_recall, threshold, binarize = binarize)
        
        if binarize:
            # assign integer values to the values above the threshold
            thresh_load = thresh_load*2
            thresh_recall = thresh_recall*3
            
            # get the overlap
            ## overlap label is assigned to voxels/vertices 
            ## where both effects are present and hence the value is 5
            ## labels are as follows:
            ## 2 - only load
            ## 3 - only recall 
            ## 5 - both load and recall
            overlap_map = thresh_recall + thresh_load
            
            # save the label file
            # create an instance of the atlas (will be used to convert data to nifti)
            atlas, a_info = am.get_atlas(atlas,gl.atlas_dir)
            nifti_img = atlas.data_to_nifti(overlap_map)
            ## save the nifti image
            nb.save(nifti_img, gl.atlas_dir + f"/tpl-SUIT/overlap_load_recall_{phase}.SUIT.nii")
            # convert to flatmap to be saved as label.gii
            img_flat = flatmap.vol_to_surf(nifti_img, space="SUIT", stats="mode")
            
            flat_gii = nt.make_label_gifti(
                                            img_flat,
                                            anatomical_struct='Cerebellum',
                                            label_names= ["none", "load", "recall", "overlap"],
                                            label_RGBA=[[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1]]
                                            )
            nb.save(flat_gii, gl.atlas_dir + f'/tpl-SUIT/overlap_load_recall_{phase}.label.gii')
            
        else:
            print("NOT IMPLEMENTED YET")
    return flat_gii

def plot_overlap_cortex(dataset = 'WMFS', 
                        type = 'CondAll',
                        ses_id = 'ses_02',
                        subject = "group",
                        atlas = 'fs32k', 
                        threshold = 80, 
                        binarize = True,
                        ):
    """get a map of overlap/unique activated areas for certain contrasts.
    1. loads in the cifti files in the cerebellar and cortical atlas spaces
    2. Calculates the desired contrast (uses the info file to get conditions)
    3. applies a threshold to the desired contrast using the percentile value provided
    4. (optional?) binarizes the thresholded maps and assigns integer values to the 
    voxels/vertices passing the threshold
    5. gets the overlap between the two contrasts
    6. (if binarized) creates a label file with labels assigned to unique and overlap regions
    """
    # create a dataset class for the dataset
    # get Dataset class for your dataset
    Data = ds.get_dataset_class(gl.base_dir, dataset=dataset)
    
    # load data 
    cifti_file = nb.load(Data.data_dir.format(subject) + f"/group_space-{atlas}_{ses_id}_{type}.dscalar.nii")
    cifti_list = nt.surf_from_cifti(cifti_file)
    # load info (will be used to select and calculate contrasts)
    info_tsv = pd.read_csv(Data.data_dir.format(subject) + f"/group_{ses_id}_info-{type}.tsv", sep="\t")
    
    # threshold and create label
    gifti_img = []
    for i, name in zip([0, 1], ['CortexLeft', 'CortexRight']):
        # get data for the hemisphere
        cifti_data = cifti_list[i]
        
        for p, phase in enumerate(["Enc", "Ret", "overall"]):
            # get the index for the phase
            if phase == "overall":
                idx = True*np.ones([len(info_tsv,)], dtype=bool)
            else:
                idx = info_tsv.phase == p
                
            info = info_tsv.loc[idx]
            data = cifti_data[idx, :]
            
            # get contrasts
            data_load, data_recall = get_contrast(data, info)
            
            # threshold the data
            thresh_load =ra.threshold_map(data_load, threshold, binarize = binarize)
            thresh_recall =ra.threshold_map(data_recall, threshold, binarize = binarize)
            
            if binarize:
                # assign integer values to the values above the threshold
                thresh_load = thresh_load*2
                thresh_recall = thresh_recall*3
                
                # get the overlap
                ## overlap label is assigned to voxels/vertices 
                ## where both effects are present and hence the value is 5
                ## labels are as follows:
                ## 2 - only load
                ## 3 - only recall 
                ## 5 - both load and recall
                overlap_map = thresh_recall + thresh_load
                # create label gifti
                gii = nt.make_label_gifti(overlap_map.T, 
                                          anatomical_struct=name,
                                          label_names= ["none", "load", "recall", "overlap"],
                                          label_RGBA=[[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1]]
                                          )
                gifti_img.append(gii)
                
            else:
                print("NOT IMPLEMENTED YET")
                
    # save gifti
    for g, hemi in enumerate(['L', 'R']):
        nb.save(gifti_img[g], gl.atlas_dir + f'/tpl-fs32k/overlap_load_recall_{phase}.{hemi}.label.gii')
    return gifti_img

if __name__=="__main__":
    plot_overlap_cerebellum()
    plot_overlap_cortex()