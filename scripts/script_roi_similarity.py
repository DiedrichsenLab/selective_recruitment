#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to make label files for the cortical and cerebellar rois

Created on 07/03/2023
Author: Ladan Shahshahani
"""
# import packages
import enum
from tabnanny import verbose
from tokenize import group
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import os
# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.util as futil
import Functional_Fusion.matrix as fmatrix
import selective_recruitment.globals as gl
import selective_recruitment.recruite_ana as ra
import selective_recruitment.plotting as splotting
import selective_recruitment.rsa as srsa
import cortico_cereb_connectivity.evaluation as ccev
import Correlation_estimation.util as corr_util
import PcmPy as pcm
import SUITPy as suit
# import surfAnalysisPy.stats as sas
import surfAnalysisPy as sa
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm, ttest_1samp
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib import pyplot as plt
import seaborn as sns
from adjustText import adjust_text # to adjust the text labels in the plots (pip install adjustText)

#
import nibabel as nb
from nilearn import plotting
import nitools as nt
import nitools.cifti as ntcifti
import SUITPy.flatmap as flatmap
from nilearn import plotting
import scipy.stats as ss

import numpy as np
from PcmPy.matrix import indicator
from scipy.linalg import solve, pinv
from scipy.spatial import procrustes
from numpy.linalg import eigh

wkdir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'


def calc_corr_per_load(atlas_space = "SUIT3", 
                        subj = None,
                        ses_id = "ses-02", 
                        smooth = True,  
                        parcellation = "NettekovenSym68c32", 
                        subtract_mean = False, 
                        type = "CondAll", 
                        verbose = False):
    """
    """
    # get datasets for all the subjects
    data,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type,  
                                    smooth = smooth, 
                                    verbose = verbose)

    # load the parcellation in 
    atlas, ainfo = am.get_atlas(atlas_space, atlas_dir=gl.atlas_dir)

    n_subj = data.shape[0]
    D_list = [] # dataframe list for subjects/loads/phases
    for s in np.arange(n_subj):
        # get data for the current subject
        data_subj = data[s, :, :]

        # loop over loads
        for l in [2, 4, 6]:
            # loop over phases
            for p, phase in enumerate(["enc", "ret"]):
                # get indices for the current phase/load - fw
                idx_fw = (info.load == l) & (info.phase == p) & (info.recall == 1)
                # get data for fw
                d_fw = data_subj[idx_fw, :]

                # get indices for the current phase/load - bw
                idx_bw = (info.load == l) & (info.phase == p) & (info.recall == 0)
                # get data for bw
                d_bw = data_subj[idx_bw, :]

                # loop over unique regions in the parcellation
                ## get the parcels
                if parcellation is not None:
                    label_img = gl.atlas_dir + '/tpl-SUIT/'+ f'atl-{parcellation}_space-SUIT_dseg.nii'
                    parcel_vec, parcels = atlas.get_parcel(label_img) 
                    parcel_vec = parcel_vec.reshape(-1, 1).T   

                    # use lookuptable to get region info
                    parcel_info = splotting.get_label_info(parcellation)               

                else: 
                    parcel_vec = np.ones([1, atlas.P])
                    parcels = [1]
                    parcel_info = ['whole']

                for p, pname in enumerate(parcel_info):
                    # get the voxels within the current parcel
                    parcel_mask = parcel_vec == p

                    # get the data within the parcel 
                    d_fw_parcel = d_fw[parcel_mask]
                    d_bw_parcel = d_bw[parcel_mask]

                    # subtract the means if chosen
                    if subtract_mean:
                        d_fw_parcel -= np.nanmean(d_fw_parcel.reshape(-1, 1), axis=0)
                        d_bw_parcel -= np.nanmean(d_bw_parcel.reshape(-1, 1), axis=0) 

                    # replace nans with 0s
                    d_fw_parcel = np.nan_to_num(d_fw_parcel)
                    d_bw_parcel = np.nan_to_num(d_bw_parcel)

                    # calculate correlation between two maps
                    R = corr_util.cosang(d_fw_parcel, d_bw_parcel)

                    # create summary dataframe
                    DD = pd.DataFrame(index = [s])
                    DD["sn"] = f"sub-{s+1:02}"
                    DD["load"] = l
                    DD["phase"] = phase
                    DD["roi"] = p
                    DD["roi_name"] = pname
                    DD["R_fwbw"] = R
                    DD["atlas"] = atlas_space

                    D_list.append(DD)
    return pd.concat(D_list, ignore_index=True)


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


def test_parcellation(atlas_space = "SUIT3", label = "NettekovenSym68c32integLR"): 
    """
    Testing parcellations created with integrate_subparcels
    """
    Nii = nb.load(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii")
    data = suit.vol_to_surf(Nii,space='SUIT', stats="mode")
    

    # get the lookuptable
    idx_lable, colors, lablenames = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")
    # adding 0
    lablenames.insert(0, '0')
    # adding color for 0
    color0 = np.array([0, 0, 0])
    colors = np.vstack([color0, colors])
    colors = np.hstack([colors, np.ones([colors.shape[0], 1])])

    cmap = LinearSegmentedColormap.from_list("my_colors", colors)

    gii = suit.flatmap.make_label_gifti(
                    data,
                    anatomical_struct='Cerebellum',
                    label_names=lablenames,
                    # column_names=[],
                    label_RGBA=colors
                    )

    nb.save(gii, "test.label.gii")

    return


def est_G(Y, X=None, S=None):
    """
    Obtains a crossvalidated estimate of G
    Y = Z @ U + X @ B + E, where var(U) = G

    Parameters:
        Y (numpy.ndarray)
            Activity data
        Z (numpy.ndarray)
            2-d: Design matrix for conditions / features U
            1-d: condition vector
        part_vec (numpy.ndarray)
            Vector indicating the partition number
        X (numpy.ndarray)
            Fixed effects to be removed
        S (numpy.ndarray)

    Returns:
        G_hat (numpy.ndarray)
            n_cond x n_cond matrix
        Sig (numpy.ndarray)
            n_cond x n_cond noise estimate per block

    """

    n_channel , n_cond = Y.shape
    G = Y @ Y.T 

    return G


def get_region_info(label = 'NettekovenSym68c32AP'):
    # get the roi numbers of Ds only
    idx_label, colors, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")
    
    
    D_indx = [label_names.index(name) for name in label_names if "D" in name]
    D_name = [name for name in label_names if "D" in name]

    # get the colors of Ds
    colors_D = colors[D_indx, :]

    D_list = []
    for rr, reg in enumerate(D_indx):
        reg_dict = {}
        reg_dict['region'] = reg
        reg_dict['region_name'] = D_name[rr]
        reg_dict['region_id'] = rr
        if 'L' in label_names[reg]:
            reg_dict['side'] = 'L'
        if 'R' in label_names[reg]:
            reg_dict['side'] = 'R'
        if 'A' in label_names[reg]:
            reg_dict['anterior'] = True
        if 'P' in label_names[reg]:
            reg_dict['anterior'] =False
        D_list.append(pd.DataFrame(reg_dict, index=[rr]))
    Dinfo = pd.concat(D_list)

    return Dinfo, D_indx, colors_D


def calc_G_group(center = False, reorder = ['side', 'anterior']):
    """
    creating MDS plots with conditions as axes
    """
    # TODO: Goal: To investigate the activity profiles of different regions. In other words, how similar those regions are.
    # TODO: Get the similarity between regions. Use the variance covariance between activity profiles of regions wthin subjects
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")

    # get the data tensor
    tensor, info, _ = ds.get_dataset(gl.base_dir,"WMFS",atlas="SUIT3",sess="ses-02",type='CondAll', info_only=False)

    # create atlas object
    atlas_suit, _ = am.get_atlas("SUIT3",gl.atlas_dir)

    # make the label name for cerebellar parcellation
    lable_file = f"{gl.atlas_dir}/tpl-SUIT/atl-NettekovenSym68c32AP_space-SUIT_dseg.nii"

    # get info for D regions
    Dinfo, D_indx, colors_D = get_region_info(label = 'NettekovenSym68c32AP')
    # get parcels in the atlas

    label_vec, labels = atlas_suit.get_parcel(lable_file)
    # average data within parcels
    ## dimensions will be #subjects-by-#numberCondition-by-#region
    parcel_data, labels = ds.agg_parcels(tensor, label_vec, fcn=np.nanmean) 

    # get the parcel data corresponding to Ds
    parcel_data = parcel_data[:, :, D_indx]

    # get different dimensions of the data
    n_subj = parcel_data.shape[0]
    n_cond = parcel_data.shape[1]
    n_region = parcel_data.shape[2]
    # calculate average across subject
    data_final = np.nanmean(parcel_data, axis = 0)
    G = est_G(data_final.T)
    # pcm.plot_Gs(G[np.newaxis, ...],grid = None, labels=None)

    W, Glam = pcm.classical_mds(G,contrast=None,align=None,thres=0)

    Ginf=Dinfo.copy()
    if reorder:
        Ginf=Ginf.sort_values(reorder)
        ind=Ginf.index.to_numpy()
        G=G[ind,:][:,ind]
        Ginf=Ginf.reset_index()
    

    return G, W, Glam, Ginf, colors_D

def calc_G(center = False, 
           subj = None, 
           label = 'NettekovenSym68c32AP', 
           reorder = ['side', 'anterior']):
    """
    """
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")

    # get the data tensor
    tensor, info, _ = ds.get_dataset(gl.base_dir, subj = subj,
                                     dataset="WMFS",atlas="SUIT3",
                                     sess="ses-02",type='CondAll', info_only=False)

    # create atlas object
    atlas_suit, _ = am.get_atlas("SUIT3",gl.atlas_dir)

    # make the label name for cerebellar parcellation
    lable_file = f"{gl.atlas_dir}/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii"

    # get info for D regions
    Dinfo, D_indx, colors_D = get_region_info(label = label)
    # get parcels in the atlas

    label_vec, labels = atlas_suit.get_parcel(lable_file)
    # average data within parcels
    ## dimensions will be #subjects-by-#numberCondition-by-#region
    parcel_data, labels = ds.agg_parcels(tensor, label_vec, fcn=np.nanmean) 

    # get the parcel data corresponding to Ds
    parcel_data = parcel_data[:, :, D_indx]

    # get different dimensions of the data
    n_subj = parcel_data.shape[0]
    n_cond = parcel_data.shape[1]
    n_region = parcel_data.shape[2]

    # loop over subjects and estimate non-cross-validated G
    G = np.zeros([n_subj, parcel_data.shape[2], parcel_data.shape[2]])
    for s in range(n_subj):
        # estimate non-cross validated G and put it inside the array
        data_subj = parcel_data[s, :, :]
        G[s, :, :] = est_G(data_subj.T)
    return G, Dinfo


if __name__ == "__main__":
    calc_G_group(center = False, reorder = ['side', 'anterior'])
    pass


