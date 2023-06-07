import selective_recruitment.globals as gl

import Functional_Fusion.atlas_map as am

import nitools as nt
import nibabel as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from pathlib import Path
from collections import OrderedDict
from matplotlib.colors import LinearSegmentedColormap

import SUITPy.flatmap as flatmap


def get_parcel_names(parcellation = "NettekovenSym68c32", atlas_space = "SUIT3"):
    """returns the list of lable names from lut file
    Args:
        parcellation (str, optional) - name of the parcellation
        atlas_space (str, optional) - name of the atlas space

    Returns:
        label_info (list) - list containing label names as they appear in the lut file
    """

    # get atlas_info
    atlas, atlas_info = am.get_atlas(atlas_dir = gl.atlas_dir, atlas_str = atlas_space)
    
    # get the lookuptable for the parcellation
    lookuptable = nt.read_lut(gl.atlas_dir + f'/{atlas_info["dir"]}/atl-{parcellation}.lut')

    # get the label info
    label_info = lookuptable[2]
    if '0' not in label_info:
        # append a 0 to it
        label_info.insert(0, '0')
    cmap = LinearSegmentedColormap.from_list("color_list", lookuptable[1])
    return label_info

def get_parcel_single(parcellation = "NettekovenSym68c32", 
                       atlas_space = "SUIT3",
                       roi_exp = "D.?R"):
    """returns a mask for the parcels that contain roi_exp

    Args:
        parcellation (str, optional): _description_. Defaults to "NettekovenSym68c32".
        atlas_space (str, optional): _description_. Defaults to "SUIT3".
        roi_exp (str, optional): _description_. Defaults to "D.?R". other examples: "D.?1.", "D.?1.|D.?2.", "D.?1.R|D.?2.R

    Returns:
        mask (boolean): list of True and Falses for where the roi_exp is found
        idx (list): list of parcel numbers for the selected parcels
        selected_ (list): list of rois that contain roi_exp
    """
    # get_label_names
    label_names = get_parcel_names(parcellation = parcellation, 
                                  atlas_space = atlas_space)
    
    # use roi_exp to get the list of rois that contain roi_exp 
    selected_ = re.findall(roi_exp, str(label_names))
    
    # make a list of boolean values for the where we find the selected_ in label_names
    mask = np.isin(label_names, selected_)
    
    # return the index number corresponding to Trues the mask
    idx = np.where(mask)[0]


    return mask, idx, selected_

def get_region_summary(label = 'NettekovenSym68c32AP', roi_super = "D"):
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

def parcels():
    
    # make path
    parcel_path = "A:\\data\\FunctionalFusion\\Atlases\\tpl-SUIT\\atl-NettekovenSym68c32_space-SUIT_probseg.nii"
    parcel_file = nb.load(parcel_path)
    
    # load in the lobules parcellation
    lobule_file = f"{gl.atlas_dir}/tpl-SUIT/atl-Anatom_space-SUIT_dseg.nii"
    
    
    # project to flatmap
    surf_data = flatmap.vol_to_surf(parcel_file, stats='nanmean',
                                     space='SUIT')
    
    lobule_data = flatmap.vol_to_surf(lobule_file, stats='mode', space='SUIT')
    
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
    
    # get the names of the labels
    idx_label, colors, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-NettekovenSym68c32.lut")
    # get the labels
    label = np.argmax(surf_data, axis=1) + 1
    
    # divide into anterior and posterior
    # loop over regions and divide them into two parts
    idx_new = [0]
    colors_new = [[0, 0, 0, 1]]
    label_new = ["0"] 
    label_array = np.zeros(label.shape)
    idx_num = 1
    for i in np.unique(label):

        # get a copy of label data
        label_copy = label.copy().reshape(-1, 1)

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
    # nii = atlas_suit.data_to_nifti(label_array)

    # # save the nifti
    # nb.save(nii, f"{gl.atlas_dir}/tpl-SUIT/atl-{label}AP_space-SUIT_dseg.nii")

    # save the lookuptable
    # nt.save_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}AP22.lut", idx_new, np.array(colors_new), label_new)
    
    gii = nt.make_label_gifti(
                    label_array.reshape(-1, 1),
                    anatomical_struct='Cerebellum',
                    labels=None,
                    label_names=label_new,
                    column_names=None,
                    label_RGBA=None
                    )
    nb.save(gii, "./test.label.gii")
    
    print("hello")
    return



if __name__ == "__main__":
    parcels()
    pass