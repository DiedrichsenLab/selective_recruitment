#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to make figures in the paper
"""

from pathlib import Path
import numpy as np

import SUITPy.flatmap as flatmap
from nilearn import plotting
import nitools as nt

import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am

import selective_recruitment.globals as gl

import cortico_cereb_connectivity.scripts.script_plot_weights as wplot


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nb



def plot_activation_map(dataset = "WMFS", 
                         ses_id = "ses-02", 
                         subj = "group",
                         type = "CondAll", 
                         atlas_space = "SUIT3", 
                         contrast_name = "average", 
                         cmap = "coolwarm",
                         cscale = [-0.2, 0.2], 
                         smooth = None):
    """
    """
    # get dataset
    data,info,dset = ds.get_dataset(gl.base_dir,
                                    dataset = dataset,
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type,  
                                    smooth = smooth)

    # get the contrast of interest
    if contrast_name == "average":
        # make up a numpy array with all 1s as we want to average all the conditions
        idx = np.ones([len(info.index), ], dtype = bool)
    else:
        idx = (info.names == contrast_name).values

    # get the data for the contrast of interest
    dat_con = np.nanmean(data[0, idx, :], axis = 0)

    # prepare data for plotting
    atlas, ainfo = am.get_atlas(atlas_space, gl.atlas_dir)
    if atlas_space == "SUIT3":
        # convert vol 2 surf
        img_nii = atlas.data_to_nifti(dat_con)
        # convert to flatmap
        img_flat = flatmap.vol_to_surf([img_nii], stats='nanmean', space = 'SUIT', ignore_zeros=True)
        ax = flatmap.plot(data=img_flat, 
                          render="plotly", 
                          hover='auto', 
                          cmap = cmap, 
                          colorbar = True, 
                          bordersize = 1, 
                          cscale = cscale)

    elif atlas_space == "fs32k":
        # get inflated cortical surfaces
        surfs = [gl.atlas_dir + f'/tpl-fs32k/tpl_fs32k_hemi-{h}_inflated.surf.gii' for i, h in enumerate(['L', 'R'])]

        # first convert to cifti
        img_cii = atlas.data_to_cifti(dat_con.reshape(-1, 1).T)
        img_con = nt.surf_from_cifti(img_cii)
        
        ax = []
        for h in [0, 1]:
            fig = plotting.plot_surf_stat_map(
                                            surfs[0], img_con[0], hemi='left',
                                            # title='Surface left hemisphere',
                                            colorbar=True, 
                                            view = 'lateral',
                                            cmap=cmap,
                                            engine='plotly',
                                            symmetric_cbar = True,
                                            vmax = cscale[1]
                                        )

            ax.append(fig.figure)
    return ax

def plot_mapwise_recruitment(data, 
                            atlas_space = "SUIT3",  
                            render = "plotly", 
                            cmap = "hsv", 
                            cscale = [-5, 5], 
                            threshold = None):
    """
    plots results of the map-wise selective recruitment on the flatmap
    """
    if threshold is not None:
        # set values outside threshold to nan
        data[np.abs(data)>threshold] = np.nan

    atlas,ainf = am.get_atlas(atlas_space, gl.atlas_dir)
    X = atlas.data_to_nifti(data)
    sdata = flatmap.vol_to_surf(X)
    fig = flatmap.plot(sdata, render=render, cmap = cmap, cscale = cscale, bordersize = 1.5)
    return fig

def plot_parcels_super(label = "NettekovenSym68c32", 
                       roi_super = "D", 
                       render = "plotly"):
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
    
    ax = flatmap.plot(roi_flat, render=render, bordersize = 1.5, 
                      overlay_type='label',
                      label_names=D_name, cmap = 'tab20b')
    return ax

def plot_parcels_single(label = "NettekovenSym68c32", 
                        roi_name = "D1R", 
                        render = "plotly"):
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
    ax = flatmap.plot(roi_flat, render=render,
                      hover='auto', colorbar = False,
                      bordersize = 1.5, overlay_type='label',
                      label_names=label_info, cmap = cmap)

    return ax

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

if __name__ == "__main__":
    plot_activation_map(dataset = "WMFS", 
                         ses_id = "ses-02", 
                         subj = "group",
                         type = "CondAll", 
                         atlas_space = "fs32k", 
                         contrast_name = "average", 
                         cmap = "coolwarm",
                         cscale = [-0.2, 0.2],  
                         smooth = None)
    pass