#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to do selective recruitment analysis

Created on 12/09/2022 at 5:25 PM
"""
# import packages
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import os
import nibabel as nb
import nitools as nt
import PcmPy as pcm
import seaborn as sb

import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import matplotlib.pyplot as plt

import selective_recruitment.rsa as srsa
import regress as sr
import selective_recruitment.globals as gl

import rsatoolbox as rsa


base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'


def cereb_cortical_rsa():
    DCereb, info, dataset = ds.get_dataset(base_dir,
                                                'WMFS',
                                                atlas='MNISymC3',
                                                sess='ses-02',
                                                type='CondRun')
    G1,Ginf = rsa.calc_rsa(DCereb,info,center=False,reorder=['phase','recall'])
    
    DCortex, info, dataset = ds.get_dataset(base_dir,
                                                'WMFS',
                                                atlas='fs32k',
                                                sess='ses-02',
                                                type='CondRun')
    G2,Ginf  = rsa.calc_rsa(DCortex,info,center=False,reorder=['phase','recall'])
    
    plt.figure(figsize=(10,10))
    pcm.vis.plot_Gs(G1)
    plt.figure(figsize=(10,10))
    pcm.vis.plot_Gs(G2)


def individual_analysis():
    DCereb, info, dataset = ds.get_dataset(base_dir,
                                                'WMFS',
                                                atlas='MNISymC3',
                                                sess='ses-02',
                                                type='CondRun')
    m1,inf = sr.calc_mean(DCereb,info,reorder=['phase','recall'])
    
    DCortex, info, dataset = ds.get_dataset(base_dir,
                                                'WMFS',
                                                atlas='fs32k',
                                                sess='ses-02',
                                                type='CondRun')
    m2,inf = sr.calc_mean(DCortex,info,reorder=['phase','recall'])
    
    D = sr.run_regress(m2,m1,inf)
    A=pd.pivot_table(D,values='Y',index='sn',columns=['phase'])
    str = [f'{i:.0f}' for i in A.index]
    plt.figure()
    sr.scatter_text(A[0],A[1],str)
    plt.xlabel('Encoding')
    plt.ylabel('Retrieval')

    B=pd.pivot_table(D,values='Y',index='sn',columns=['phase','recall'])
    enc_bf=B[0][0]-B[0][1]
    ret_bf=B[1][0]-B[1][1]
    plt.figure()
    sr.scatter_text(enc_bf,ret_bf,str)
    plt.xlabel('Encoding B-F')
    plt.ylabel('Retrieval B-F')
    pass 


def cereb_parcel_rsa(label = "NettekovenSym68c32", 
                     atlas_space = "SUIT3", 
                     subj = None,
                     type = "CondRun", 
                     label_name = "D3R", 
                     reorder = ["phase", "recall"]
                     ):

    tensor, info, dataset = ds.get_dataset(gl.base_dir,
                                           subj = subj,
                                           dataset = 'WMFS',
                                           atlas=atlas_space,
                                           sess='ses-02',
                                           type=type)

    # create atlas object to use when getting labels
    atlas, ainfo = am.get_atlas(atlas_dir=gl.atlas_dir, atlas_str=atlas_space)

    # get the label file
    label_file = f"{gl.atlas_dir}/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii"

    # read label lookup table
    idx_label, colors, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")

    # get the index for the selected label
    ## the first label is 0, discarding it ...
    i = label_names[1:].index(label_name)
    # get parcels 
    label_vector, labels = atlas.get_parcel(label_file)

    # create a mask for the voxels within the selected label
    label_mask = label_vector == i

    # loop over subject and do rsa within the selected region
    n_subj = tensor.shape[0]

    data_region = tensor[:, :, label_mask]

    G1,Ginf = srsa.calc_rsa(data_region,info,partition='run',center=False,reorder=reorder)
    return G1, Ginf

def cereb_parcel_rdm(label = "NettekovenSym68c32", 
                     atlas_space = "SUIT3", 
                     subj = None,
                     type = "CondRun", 
                     label_name = "D3R", 
                     reorder = ["phase", "recall"]
                     ):

    tensor, info, dataset = ds.get_dataset(gl.base_dir,
                                           subj = subj,
                                           dataset = 'WMFS',
                                           atlas=atlas_space,
                                           sess='ses-02',
                                           type=type)

    # create atlas object to use when getting labels
    atlas, ainfo = am.get_atlas(atlas_dir=gl.atlas_dir, atlas_str=atlas_space)

    # get the label file
    label_file = f"{gl.atlas_dir}/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii"

    # read label lookup table
    idx_label, colors, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")

    # get the index for the selected label
    ## the first label is 0, discarding it ...
    i = label_names[1:].index(label_name)
    # get parcels 
    label_vector, labels = atlas.get_parcel(label_file)

    # create a mask for the voxels within the selected label
    label_mask = label_vector == i

    # loop over subject and do rsa within the selected region
    n_subj = tensor.shape[0]

    data_region = tensor[:, :, label_mask]

    # make rsa dataset objects
    data_rsa = []
    for s in range(n_subj):
        # now create a  dataset object
        des = {'subj': s}
        obs_des = {'conds': info.cond_name.values, 'runs': info["run"].values}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(data_region.shape[2])])}
        
        # replace nans with 0s
        data_reg_tmp = np.nan_to_num(data_region[s, :, :])
        
        data_tmp = rsa.data.Dataset(measurements=data_reg_tmp,
                                    descriptors=des,
                                    obs_descriptors=obs_des,
                                    channel_descriptors=chn_des)

        data_rsa.append(data_tmp)
    # calculate crossvalidated distance for the region
    rdm_cv = rsa.rdm.calc_rdm(data_rsa, method='crossnobis', descriptor='conds', cv_descriptor='runs')
    return rdm_cv

if __name__=='__main__':
    a = cereb_parcel_rdm(label = "NettekovenSym68c32", 
                     atlas_space = "SUIT3", 
                     subj = None,
                     type = "CondRun", 
                     label_name = "D3R"
                     )

    print("hello")