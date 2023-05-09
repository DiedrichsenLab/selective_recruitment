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
import regress as ra
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

import selective_recruitment.region as sroi

import rsatoolbox as rsa

wkdir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'

# null: there is no difference between D rois 
# alternative:

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
                    parcel_info = sroi.get_label_names(parcellation)               

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


def get_region_info_all(label = 'NettekovenSym68c32AP'):
    # get the roi numbers of Ds only
    idx_label, colors, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")
    
    
    # D_indx = [label_names.index(name) for name in label_names if "D" in name]
    D_indx = [label_names.index(name) for name in label_names]
    D_name = [name for name in label_names]

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


# def calc_dist():
#     """
#     """
#     data = rsa.data.Dataset(numpy.random.rand(10, 6))
#     return

def calc_dist(center = False, 
           subj = None,
           type = "CondAll",  
           label = 'NettekovenSym68c32AP', 
           reorder = ['side', 'anterior'], 
           do_D = False):
    """
    """
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")

    # get the data tensor
    tensor, info, _ = ds.get_dataset(gl.base_dir, subj = subj,
                                     dataset="WMFS",atlas="SUIT3",
                                     sess="ses-02",type=type, info_only=False)

    # create atlas object
    atlas_suit, _ = am.get_atlas("SUIT3",gl.atlas_dir)

    # make the label name for cerebellar parcellation
    lable_file = f"{gl.atlas_dir}/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii"

    
    # get parcels in the atlas
    label_vec, labels = atlas_suit.get_parcel(lable_file)
    # average data within parcels
    ## dimensions will be #subjects-by-#numberCondition-by-#region
    parcel_data, labels = ds.agg_parcels(tensor, label_vec, fcn=np.nanmean) 

    # get the parcel data corresponding to Ds
    # get info for D regions
    Dinfo, D_indx, colors_D = get_region_info(label = label)
    parcel_data = parcel_data[:, :, D_indx]

    # data = rsa.data.Dataset(np.random.rand(10, 6)) # 10 random observations of 6 channels

    # get different dimensions of the data
    n_subj = parcel_data.shape[0]
    n_cond = parcel_data.shape[1]
    n_region = parcel_data.shape[2]

    # create rsa dataset objects 
    data_rsa = []
    for s in range(n_subj):
        des = {'session': 1, 'subj': s}
        chn_des = {'conds': info.cond_name.values}
        obs_des = {'voxels': Dinfo.region_name.values}

        data_rsa.append(rsa.data.Dataset(measurements=parcel_data[s, :, :].T,
                                descriptors=des,
                                obs_descriptors=obs_des,
                                channel_descriptors=chn_des))

    # calculate euclidean distance for each subject
    ## distances between the activity profiles of subregions of D
    RDM_euc = rsa.rdm.calc_rdm(data_rsa, descriptor='voxels')
    rdms_ = rsa.rdm.RDMs(RDM_euc.dissimilarities,
                rdm_descriptors={'name': np.array([f"RDM{i}"
                                                    for i in range(RDM_euc.n_rdm)])}
                )

    fig, ax, ret_val = rsa.vis.show_rdm(RDM_euc)
    # rsa.vis.rdm_comparison_scatterplot(rdms_)
    return 

def calc_G(center = False, 
           subj = None,
           type = "CondAll",  
           label = 'NettekovenSym68c32AP', 
           reorder = ['side', 'anterior'], 
           do_D = False):
    """
    """
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")

    # get the data tensor
    tensor, info, _ = ds.get_dataset(gl.base_dir, subj = subj,
                                     dataset="WMFS",atlas="SUIT3",
                                     sess="ses-02",type=type, info_only=False)

    # create atlas object
    atlas_suit, _ = am.get_atlas("SUIT3",gl.atlas_dir)

    # make the label name for cerebellar parcellation
    lable_file = f"{gl.atlas_dir}/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii"

    
    # get parcels in the atlas
    label_vec, labels = atlas_suit.get_parcel(lable_file)
    # average data within parcels
    ## dimensions will be #subjects-by-#numberCondition-by-#region
    parcel_data, labels = ds.agg_parcels(tensor, label_vec, fcn=np.nanmean) 

    # get the parcel data corresponding to Ds
    if do_D:
        # get info for D regions
        Dinfo, D_indx, colors_D = sroi.get_region_info(label = label)
        parcel_data = parcel_data[:, :, D_indx]
    else:
        # get info for D regions
        Dinfo, D_indx, colors_D = get_region_info_all(label = label)
        Dinfo = Dinfo[1:]

    # get different dimensions of the data
    n_subj = parcel_data.shape[0]
    n_cond = parcel_data.shape[1]
    n_region = parcel_data.shape[2]

    # loop over subjects and estimate non-cross-validated G
    G = np.zeros([n_subj, parcel_data.shape[2], parcel_data.shape[2]])
    G_vec = np.zeros([n_subj, int((n_region*(n_region-1))/2)+n_region])
    for s in range(n_subj):
        # estimate non-cross validated G and put it inside the array
        data_subj = parcel_data[s, :, :].T
        n_channel , n_cond = data_subj.shape
        
        # Gs are symmetric
        G_tmp = data_subj @ data_subj.T 

        # get the upper triangular part
        triu_idx = np.triu_indices(n_region, k = 0)
        G_vec[s, :] = G_tmp[triu_idx]

        G[s, :, :] = G_tmp
    return G, G_vec

def calc_G_cv(center = False, 
              subj = None, 
              label = 'NettekovenSym68c32AP',
              type = "CondRun"):
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")

    # get the data tensor
    tensor, info, _ = ds.get_dataset(gl.base_dir, subj = subj,
                                     dataset="WMFS",atlas="SUIT3",
                                     sess="ses-02",type=type, info_only=False)

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

    # get z
    regions = Dinfo.region.values
    Z = pcm.matrix.indicator(Dinfo.region)
    # get partition vector
    part_vec = np.unique(info.run.values)

    # loop over subjects and estimate non-cross-validated G
    G = np.zeros([n_subj, parcel_data.shape[2], parcel_data.shape[2]])
    for s in range(n_subj):
        # estimate non-cross validated G and put it inside the array
        data_subj = parcel_data[s, :, :].T

        # need to rearrange data

        G_cv = pcm.est_G_crossval(data_subj, Z, part_vec, X=None, S=None)
        G[s, :, :] = G_cv

    return
if __name__ == "__main__":
    calc_dist(center = False, 
                        subj = None, 
                        label = 'NettekovenSym68c32AP', 
                        do_D = True)

    print("hello")
    pass


