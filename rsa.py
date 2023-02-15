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

def cossim(G1,G2,axis = None):
    """Calculates the cosine similarity between two matrices 

    Args:
        G1 (ndarray): matrix 1
        G2 (ndarray): matrix 2
    """
    if axis is None: 
        c = np.sum(G1*G2)
        a = np.sum(G1*G1)
        b = np.sum(G2*G2)
        return c/np.sqrt(a*b)
    elif axis == 1:
        c = G1@G2.T
        a = np.sum(G1*G1,axis=1)
        b = np.sum(G2*G2,axis=1)
        return c/np.sqrt(np.outer(a,b))
    else:
        raise(NameError('axis must be 1 or none'))

def sim_difference_mean(G1,G2,sim_measure=cossim):
    """ Does a correspondence analysis to check if there 
    are systematic difference between two sets of vectors / matrices or tensors
    For each subject it calculates the similarity measure between G1[i] and the mean of G1 and G2 (excluding subject i)
    This is averaged with the same similarity measure, now reversing the role of G1 and G2
    Args:
        G1 (ndarray): nsubj x ... tensor of data 1 
        G2 (ndarray): nsubj x ... tensor of dayta 2
        sim_measure (fcn or str): Function 
    Returns: 
        results (ndarray): nsubj x 2. First column is similarity with itself
                            second column is with the opposite. 
                            test results[:,0]>results[:,1]
    """ 
    if G1.shape !=G2.shape:
        raise(NameError('Tensors G1 and G2 need to have the same size'))
    nsubj = G1.shape[0]
    result = np.zeros((nsubj,4))
    indx = np.arange(nsubj)
    for i in indx:
        result[i,0]=sim_measure(G1[i],np.mean(G1[indx!=i],axis=0))
        result[i,1]=sim_measure(G1[i],np.mean(G2[indx!=i],axis=0))
        result[i,2]=sim_measure(G2[i],np.mean(G2[indx!=i],axis=0))
        result[i,3]=sim_measure(G2[i],np.mean(G1[indx!=i],axis=0))
    return (result[:,:2]+result[:,2:])/2

def mmd_unbiased_paired(G1,G2,sim_measure=cossim):
    """ Calculates the unbiased Maximum mean divergence2
        for a paired group of samples (m=n)
        G1 (ndarray): nsubj x ... tensor of data 1 
        G2 (ndarray): nsubj x ... tensor of data 2
        sim_measure (fcn or str): Kernel function 
    Returns: 
        mmd2 (ndarray): 

    """ 
    if G1.shape !=G2.shape:
        raise(NameError('Tensors G1 and G2 need to have the same size'))
    nsubj = G1.shape[0]
    G1 = G1.reshape(nsubj,-1)
    G2 = G2.reshape(nsubj,-1)

    C11 = sim_measure(G1,G1,axis=1)
    C12 = sim_measure(G1,G2,axis=1)
    C22 = sim_measure(G2,G2,axis=1)
    k11 = C11.sum() - np.trace(C11) 
    k22 = C22.sum() - np.trace(C22)
    k12 = C12.sum() - np.trace(C12)

    mmd2 = 1/(nsubj * (nsubj-1)) * (k11 + k22 - 2*k12)
    return mmd2