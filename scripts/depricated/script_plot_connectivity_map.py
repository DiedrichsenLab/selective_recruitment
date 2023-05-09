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
import regress as ra
import selective_recruitment.plotting as splotting
import selective_recruitment.rsa as srsa

import cortico_cereb_connectivity.evaluation as ccev
import cortico_cereb_connectivity.scripts.script_plot_weights as wplot

import Correlation_estimation.util as corr_util
import SUITPy as suit
# import surfAnalysisPy.stats as sas
import surfAnalysisPy as sa
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
#
import nibabel as nb
import nitools as nt
import SUITPy.flatmap as flatmap
from nilearn import plotting


def plot_connectivity_weight(roi_name = "D2R",
                             method = "L2Regression",
                             cortex_roi = "Icosahedron1002",
                             cerebellum_roi = "NettekovenSym68c32",
                             cerebellum_atlas = "SUIT3",
                             log_alpha = 8,
                             dataset_name = "MDTB",
                             cmap = "coolwarm",
                             ses_id = "ses-s1"):
    """
    """
    # get connectivity weight maps for the selected region
    cifti_img = wplot.get_weight_map(method = method,
                                    cortex_roi = cortex_roi,
                                    cerebellum_roi = cerebellum_roi,
                                    cerebellum_atlas = cerebellum_atlas,
                                    log_alpha = log_alpha,
                                    dataset_name = dataset_name,
                                    ses_id = ses_id,
                                    type = "dscalar"
                                    )

    # get the cortical weight map corresponding to the current
    ## get parcel axis from the cifti image
    parcel_axis = cifti_img.header.get_axis(0)
    ## get the name of the parcels in the parcel_axis
    idx = list(parcel_axis.name).index(roi_name)
    # get the maps for left and right hemi
    weight_map_list = nt.surf_from_cifti(cifti_img)
    # get the map for the selected region for left and right hemispheres
    weight_roi_list = [weight_map_list[h][idx, :] for h in [0, 1]]

    surf_hemi = []
    fig_hemi = []
    for h, hemi in enumerate(['L', 'R']):
        img = weight_map_list[h]
        # get the numpy array corresponding to the contrast
        img_data = weight_roi_list[h]
        surf_hemi.append(gl.atlas_dir + f"/tpl-fs32k/tpl_fs32k_hemi-{hemi}_inflated.surf.gii")

        fig_hemi.append(plotting.view_surf(
                                        surf_hemi[h], img_data, colorbar=True,
                                        cmap=cmap, vmax = np.nanmax(img_data),
                                        vmin = np.nanmin(img_data)
                                        ))
    return fig_hemi