import selective_recruitment.globals as gl

import Functional_Fusion.atlas_map as am

import nitools as nt
import nibabel as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import OrderedDict
from matplotlib.colors import LinearSegmentedColormap

def make_parcelfs32k_lut(atlas_space = "fs32k", label = "glasser"):
    """
    makes a lut file for the parcellation in fs32k based on the label.gii file
    Args:
        atlas_space (str) - default is "fs32k"
        label (str) - name of the parcellation label.gii file saved in atl-fs32k directory
    """

    # load in the parcellation label file
    label_file = [nb.load(f"{gl.atlas_dir}/tpl-fs32k/{label}.{hemi}.label.gii") for hemi in ["L", "R"]]

    # get the color rgba values: 0 will be discarded
    ## get_gifti_colortable returns rgba values and cmap object.
    ## we only need the first item it returns - > index = 0
    color_list = [nt.get_gifti_colortable(label_file[h],ignore_zero=False)[0][1:] for h in [0, 1]]

    # get gifti labels: 0 (???) will be discarded
    ## get_gifti_labels returns a list, it needs to be converted to a numpy array
    label_names = [np.array(nt.get_gifti_labels(label_file[h])[1:]) for h in [0, 1]]

    # concatenate colors and label names for left and right
    colors = np.concatenate(color_list, axis = 0)
    labels = np.concatenate(label_names, axis = 0)
    indices = np.concatenate([np.arange(1, len(label_names[h])+1) for h in [0, 1]], axis = 0)

    # make the lut file
    fname = f"{gl.atlas_dir}/tpl-fs32k/atl-{label}.lut"
    nt.save_lut(fname,indices,colors,labels)
    return

def make_glasser_select_list():
    """
    get information from glasser lut for the selected regions
    """
    # hard-coded list: These rois from left and right hemi will be fetched
    sub_roi_list = ['6v', 'a9_46v', '46', 
                    'IFJp', '6r', 'LIPv', 
                    'MIP', 'IPS1', 'AIP', 
                    'PFt', 'FST', 'POP4', 
                    'SCEF', '7PL', 'v4', 
                    'LO2', 'FEF', '6a']
    index, colors, labels= nt.read_lut(fname = f"{gl.atlas_dir}/tpl-fs32k/atl-glasser.lut")

    # get the indices for the selected regions
    roi_left = [f"L_{name}_ROI" for name in sub_roi_list]
    roi_right = [f"R_{name}_ROI" for name in sub_roi_list]
    roi_list = [roi_left, roi_right]
    struct_names = ['CortexLeft', 'CortexRight']
    # create label files for left and right separately
    for h, hemi in enumerate(['L', 'R']):
        # select a mask for the current hemisphere
        selected_mask = np.isin(labels, roi_list[h])

        # use the mask to get the info for the selected regions
        index_sel = index[selected_mask]
        colors_sel = colors[selected_mask]
        labels_sel = np.array(labels)[selected_mask]

        colors_sel = np.hstack([colors_sel, np.ones([colors_sel.shape[0], 1])])

        # make label files to test
        ## first load in the old label file
        label_old = nb.load(f"{gl.atlas_dir}/tpl-fs32k/glasser.{hemi}.label.gii")

        # get data
        old_dat = label_old.agg_data()

        ## make a mask for selected regions
        gii_mask = np.isin(old_dat, index_sel)
        ## get selected labels/vertices
        labels_new = old_dat.copy()
        labels_new[np.logical_not(gii_mask)] = 0

        labels_sel = np.hstack(["none", labels_sel])

        colors_sel = np.vstack([[0, 0, 0, 1], colors_sel])

        
        # save new giftis
        gii = nt.make_label_gifti(
                    labels_new.astype(int).reshape(-1, 1),
                    anatomical_struct=struct_names[h],
                    labels=None,
                    label_names=labels_sel,
                    column_names=None,
                    label_RGBA=colors_sel
                    )
        nb.save(gii,f"{gl.atlas_dir}/tpl-fs32k/glasser_selected.{hemi}.label.gii")
    return