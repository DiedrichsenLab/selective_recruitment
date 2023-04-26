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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
import seaborn as sns
from adjustText import adjust_text # to adjust the text labels in the plots (pip install adjustText)

#
import nibabel as nb
import nitools as nt
import SUITPy.flatmap as flatmap

def plot_parcels_super(label = "NettekovenSym68c32", roi_super = "D"):
    """
    Plots the selected super region based on the parcellation defined by "parcellation"
    Args:
        parcellation (str) - name of the hierarchical parcellation 
        roi_super (str) - name assigned to the roi in the hierarchical parcellation
    Returns:
        None
    """
    # get the roi numbers for the super roi
    idx_label, colors2, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")
    
    D_indx = [label_names.index(name) for name in label_names if roi_super in name]
    D_name = [name for name in label_names if roi_super in name]
    D_name.insert(0, '0')

    fname = gl.atlas_dir + f'/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii'
    img = nb.load(fname)
    dat = img.get_fdata()
    
    # map it from volume to surface
    img_flat = flatmap.vol_to_surf([img], stats='mode', space = 'SUIT')

    # get a mask for the selected super region
    region_mask = np.isin(img_flat.astype(int), D_indx)

    # convert non-selected labels to nan
    roi_flat = img_flat.copy()
    # convert non-selected labels to nan
    roi_flat[np.logical_not(region_mask)] = np.nan

    for i, r in enumerate(D_indx):
        roi_flat[roi_flat == r] = i+1
    
    ax = flatmap.plot(roi_flat, render="plotly", bordersize = 1.5, 
                      overlay_type='label',
                      label_names=D_name, cmap = 'tab20b')
    return ax

def plot_parcels_single(label = "NettekovenSym68c32", roi_name = "D1R"):
    """
    plot the selected region from parcellation on flatmap
    Args:
        parcellation (str) - name of the parcellation
        roi_name (str) - name of the roi as stored in the lookup table
    Return:
        ax (axes object)
        roi_num (int) - number corresponding to the region
    """
    fname = gl.atlas_dir + f'/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii'
    img = nb.load(fname)
    # map it from volume to surface
    img_flat = flatmap.vol_to_surf([img], stats='mode', space = 'SUIT')

    # get the lookuptable for the parcellation
    lookuptable = nt.read_lut(gl.atlas_dir + f'/tpl-SUIT/atl-{label}.lut')

    # get the label info
    label_info = lookuptable[2]
    if '0' not in label_info:
        # append a 0 to it
        label_info.insert(0, '0')
    cmap = LinearSegmentedColormap.from_list("color_list", lookuptable[1])

    # get the index for the region
    roi_num = label_info.index(roi_name)
    roi_flat = img_flat.copy()
    # convert non-selected labels to nan
    roi_flat[roi_flat != float(roi_num)] = np.nan
    # plot the roi
    ax = flatmap.plot(roi_flat, render="plotly",
                      hover='auto', colorbar = False,
                      bordersize = 1.5, overlay_type='label',
                      label_names=label_info, cmap = cmap)

    return ax

if __name__ == "__main__":
    pass
