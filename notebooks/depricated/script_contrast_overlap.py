# TODO: write code to plot the overlap between contrasts and plot on surface of cortex and cerebellum (Volume and flatmap)
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
import sys

sys.path.append('../cortico-cereb_connectivity')
sys.path.append('../Functional_Fusion') 
sys.path.append('..')

# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import selective_recruitment.globals as gl
import selective_recruitment.recruite_ana as ra

#
import os
import nibabel as nb
import nitools as nt

def make_roi_label(dataset_name = "WMFS", 
                   ses_id = "ses-s1", 
                   threshold = 80, 
                   binarize = True):
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
    cifti_cerebellum = nb.load(Data.data_dir.format("group") + f"/group_space-SUIT3_{ses_id}_CondAll.dscalar.nii")
    cifti_cortex = nb.load(Data.data_dir.format("group") + f"/group_space-fs32k_{ses_id}_CondAll.dscalar.nii")
    
    # load info (will be used to select contrast)
    info_tsv = pd.read_csv(Data.data_dir.format("group") + f"/group_{ses_id}_info-CondAll.tsv", sep="\t")

    # get the effects of interest in encoding and retrieval
    for phase in [0, 1]:
        ## get the indices for the current phase
        idx = info_tsv["phase"] == phase
        ## get load effect

        ## get recall effect


    # # save the nifti image
    # nb.save(roi_nifti, Data.atlas_dir + '/tpl-SUIT' + f'/atl-{contrast}_space-SUIT_dseg.nii')
    
    # # save label gifti images
    # for i, h in enumerate(['L', 'R']):
    #     nb.save(roi_gifti[i], Data.atlas_dir + '/tpl-fs32k' + f'/{contrast}.32k.{h}.label.gii')

    return 


if __name__ == "__main__":
    make_roi_label(dataset_name = "WMFS", 
                   ses_id = "ses-s1", 
                   threshold = 80, 
                   binarize = True)

