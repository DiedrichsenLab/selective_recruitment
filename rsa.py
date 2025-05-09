#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to do RSA component of the selective recruitment analysis
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
    """ Calculates crossvalidated second moment matrices across a number  
    of subjects. Option to center (voxel-mean-subtract) and to reorder the conditions 
    Args:
        data (ndarray): nsubj x nobs x nvox tensor of data 
        info (_type_): nobs data frame with information 
        partition (str): Column name of partition indicator. Defaults to 'run'.
        center (bool): Subtract mean of each voxel? Defaults to False.
        reorder (False or List): If str of list of str, reorders conditions based on those columns .

    Returns:
        G (ndarray): nsubj x ncond x ncond tensor of second moments
        Ginf (pd.DataFrame): information on different conditions 
    """
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
    
    Z = pcm.matrix.indicator(info.reg_id)
    
    for i in range(n_subj):
        G[i],_=pcm.est_G_crossval(data[i],Z,info[partition],X)
    
    part = np.unique(info[partition])
    Ginf=info[info[partition]==part[0]].copy()
    if reorder:
        Ginf=Ginf.sort_values(reorder)
        ind=Ginf.index.to_numpy()
        G=G[:,ind,:][:,:,ind]
        Ginf=Ginf.reset_index()
    
    return G,Ginf

def test_rsa_difference(G1,G2): 
    pass 

