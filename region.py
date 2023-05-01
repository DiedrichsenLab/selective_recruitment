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

def get_label_names(parcellation):
    # get the lookuptable for the parcellation
    lookuptable = nt.read_lut(gl.atlas_dir + f'/tpl-SUIT/atl-{parcellation}.lut')

    # get the label info
    label_info = lookuptable[2]
    if '0' not in label_info:
        # append a 0 to it
        label_info.insert(0, '0')
    cmap = LinearSegmentedColormap.from_list("color_list", lookuptable[1])
    return label_info

def get_region_info(label = 'NettekovenSym68c32AP', roi_super = "D"):
    # get the roi numbers of Ds only
    idx_label, colors, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")
    
    
    D_indx = [label_names.index(name) for name in label_names if roi_super in name]
    D_name = [name for name in label_names if roi_super in name]

    # get the colors of Ds
    colors_D = colors[D_indx, :]

    D_list = []
    for rr, reg in enumerate(D_indx):
        reg_dict = {}
        reg_dict['roi'] = reg
        reg_dict['roi_name'] = D_name[rr]
        reg_dict['roi_id'] = rr
        if 'L' in label_names[reg]:
            reg_dict['roi_side'] = 'L'
        if 'R' in label_names[reg]:
            reg_dict['roi_side'] = 'R'
        if 'A' in label_names[reg]:
            reg_dict['roi_AP'] = 'A'
        if 'P' in label_names[reg]:
            reg_dict['roi_AP'] ='P'
        D_list.append(pd.DataFrame(reg_dict, index=[rr]))
    Dinfo = pd.concat(D_list)

    return Dinfo, D_indx, colors_D

def divide_by_horiz(atlas_space = "SUIT3", label = "NettekovenSym68c32"):
    """
    Create a mask that divides the cerebellum into anterior and posterior
    demarkated by horizontal fissure
    And then creates a new label file (alongside lookup table) with parcels divided by
    horizontal fissure
    Args:
        atlas_space (str) - string representing the atlas space
    Returns:
        mask_data (np.ndarray)
        mask_nii (nb.NiftiImage)
    """
    # create an instance of atlas object
    atlas_suit, ainfo = am.get_atlas(atlas_space, gl.atlas_dir)

    # load in the lobules parcellation
    lobule_file = f"{gl.atlas_dir}/tpl-SUIT/atl-Anatom_space-SUIT_dseg.nii"
    
    # get the lobules in the atlas space
    lobule_data, lobules = atlas_suit.get_parcel(lobule_file)

    # load the lut file for the lobules 
    idx_lobule, _, lobule_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-Anatom.lut")

    # demarcate the horizontal fissure
    ## horizontal fissure is between crusI and crusII.
    ## everything above crusII is the anterior part
    ## find the indices for crusII
    crusII_idx = [lobule_names.index(name) for name in lobule_names if "CrusII" in name]
    posterior_idx = idx_lobule[min(crusII_idx):]
    anterior_idx = idx_lobule[0:min(crusII_idx)]

    # assign value 1 to the anterior part
    anterior_mask = np.isin(lobule_data, anterior_idx)

    # assign value 2 to the posterior part
    posterior_mask = np.isin(lobule_data, posterior_idx)

    # get the label file
    lable_file = f"{gl.atlas_dir}/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii"
    # load the lut file for the label  
    idx_label, colors, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")

    # get the parcels in the atlas space
    label_data, labels = atlas_suit.get_parcel(lable_file)

    # loop over regions and divide them into two parts
    idx_new = [0]
    colors_new = [[0, 0, 0, 1]]
    label_new = ["0"] 
    label_array = np.zeros(label_data.shape)
    idx_num = 1
    for i in labels:

        # get a copy of label data
        label_copy = label_data.copy()

        # convert all the labels other than the current one to NaNs
        label_copy[label_copy != i] = 0

        # get the anterior and posterior part
        label_anterior = label_copy * anterior_mask
        label_posterior = label_copy * posterior_mask

        if any(label_anterior): # some labels only have posterior parts
            # get the anterior part
            label_new.append(f"{label_names[i]}_A")
            idx_new.append(idx_num)
            colors_new.append(list(np.append(colors[i, :], 1))) 
            label_array[np.argwhere(label_anterior)] = idx_num
            idx_num=idx_num+1

        if any(label_posterior):
            # get the posterior part
            label_new.append(f"{label_names[i]}_P")
            idx_new.append(idx_num)
            colors_new.append(list(np.append(colors[i, :], 1))) 
            label_array[np.argwhere(label_posterior)] = idx_num

            idx_num=idx_num+1


    # create a nifti object
    nii = atlas_suit.data_to_nifti(label_array)

    # save the nifti
    nb.save(nii, f"{gl.atlas_dir}/tpl-SUIT/atl-{label}AP_space-SUIT_dseg.nii")

    # save the lookuptable
    nt.save_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}AP.lut", idx_new, np.array(colors_new), label_new)
    return nii

def integrate_subparcels(atlas_space = "SUIT3", label = "NettekovenSym68c32", LR = False):
    """ Integrates subparcels together and create a new one
        For example, it puts together all the Ds and create one single D parcel

    """
    # create an instance of atlas object
    atlas_suit, ainfo = am.get_atlas(atlas_space, gl.atlas_dir)
    # get the label file
    lable_file = f"{gl.atlas_dir}/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii"
    # load the lut file for the label  
    idx_label, colors, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")

    # get the parcels in the atlas space
    label_data, labels = atlas_suit.get_parcel(lable_file)

    # integrating over labels with same names
    # get unique starting letters
    # these will be the new set of roi names for the integrated parcellation
    if LR: # if you want left and right separated
        l0_all = [f"{name[0]}{hemi}" for name in label_names[1:] for hemi in ['L', 'R']] # ignoring the first one which is 0
        label_names_new = list(OrderedDict.fromkeys(l0_all))
        ## get indices of old labels starting with new labels
        labels_idx = []
        for letter in label_names_new:
            labels_idx.append([idx_label[i] for i, ltr in enumerate(label_names) if (ltr.startswith(letter[0])) & (ltr.endswith(letter[-1]))])
        fname = f"{label}integLR"
    else:
        l0_all = [name[0] for name in label_names[1:]] # ignoring the first one which is 0
        label_names_new = list(OrderedDict.fromkeys(l0_all))
        labels_idx = []
        for letter in label_names_new:
            labels_idx.append([idx_label[i] for i, ltr in enumerate(label_names) if ltr.startswith(letter)])
        fname = f"{label}integ"

    # label_names_new.insert(0, '0')
    # loop over these new labels and get the indices
    # re-number the parcels in label_data
    label_data_new = np.zeros(label_data.shape)
    ## get indices of old labels starting with new labels

    idx_new = []
    for lid, lname in enumerate(label_names_new):
        print(f"{lid} {lname}")
        # get a mask for the current label
        label_mask = np.isin(label_data, labels_idx[lid])

        # use the mask to set new labels
        label_data_new[label_mask] = lid+1
        idx_new.append(lid+1)

    # create a nifti object
    nii = atlas_suit.data_to_nifti(label_data_new)

    # save the nifti
    nb.save(nii, f"{gl.atlas_dir}/tpl-SUIT/atl-{fname}_space-SUIT_dseg.nii")

    # create colors:
    cmap = plt.cm.get_cmap("hsv", len(np.unique(label_data_new)[1:]))
    colors_new = cmap(range(len(np.unique(label_data_new)[1:])))

    # save the lookuptable
    nt.save_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{fname}.lut", idx_new, np.array(colors_new), label_names_new)
    return
