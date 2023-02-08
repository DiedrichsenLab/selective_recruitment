#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to do selective recruitment analysis

Created on 12/09/2022 at 5:25 PM
Author: Ladan Shahshahani
"""
# import packages
import sys

import numpy as np
import pandas as pd
from pathlib import Path

# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.atlas_map as ds

import os
import nibabel as nb
import nitools as nt
import PcmPy as pcm 


def calc_rsa(data,info,partition='run',center=False,reorder=False):
    n_subj = data.shape[0]
    cond=np.unique(info.reg_id)
    n_cond = len(cond)

    # For this purpose, ignore nan voxels
    data = np.nan_to_num(data,copy=False)
    G = np.zeros((n_subj,n_cond,n_cond))
    
    # Subtract the mean or express relative to rest?
    if center: 
        X = pcm.matrix.indicator(info[partition])
    else:
        X = None
    
    mean_d = data.mean(axis=2)
    Z = pcm.matrix.indicator(info.reg_id)
    mean_d = mean_d @ np.linalg.pinv(Z).T
    
    for i in range(n_subj):
        G[i],_=pcm.est_G_crossval(data[i],Z,info[partition],X)
    
    part = np.unique(info[partition])
    Ginf=info[info[partition]==part[0]].copy()
    if reorder:
        Ginf=Ginf.sort_values(reorder)
        ind=Ginf.index.to_numpy()
        G=G[:,ind,:][:,:,ind]
        Ginf=Ginf.reset_index()
        mean_d = mean_d[:,ind]
    
    return G,Ginf,mean_d


