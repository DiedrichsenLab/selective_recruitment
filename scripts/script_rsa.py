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
import scipy.stats as ss

# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import matplotlib.pyplot as plt
# modules from connectivity
import cortico_cereb_connectivity.prepare_data as cprep
import selective_recruitment.rsa as rsa
import selective_recruitment.recruite_ana as sr


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
    G1,Ginf = rsa.calc_rsa(DCereb,info,center=True,reorder=['phase','recall'])
    
    DCortex, info, dataset = ds.get_dataset(base_dir,
                                                'WMFS',
                                                atlas='fs32k',
                                                sess='ses-02',
                                                type='CondRun')
    G2,Ginf  = rsa.calc_rsa(DCortex,info,center=True,reorder=['phase','recall'])
    
    plt.figure(figsize=(10,10))
    pcm.vis.plot_Gs(G1)
    plt.figure(figsize=(10,10))
    pcm.vis.plot_Gs(G2)

    d1 = rsa.sim_difference_mean(G1,G2)
    res1 = ss.ttest_rel(d1[:,0], d1[:,1])
    print(res1)
    pass


def test_test_sim_differences_type1(s1=1,s2=1,nsubj=20,niter=1000,P =9):
    """Test for type I error rate in a case where the 
    variability of X1 and X2 is different. 
    """
    m = np.random.normal(0,1,(P,1))
    p = np.zeros((niter,))
    t = np.zeros((niter,))
    for i in range(niter):
        X1 = np.random.normal(0,s1,(nsubj,) + m.shape) + m
        X2 = np.random.normal(0,s2,(nsubj,) + m.shape) + m
        d = rsa.sim_difference_mean(X1,X2)
        t[i],p[i] = ss.ttest_rel(d[:,0], d[:,1],alternative='greater')

    plt.subplot(1,3,1)
    sb.histplot(t)
    plt.axvline(t.mean(),color='k')

    plt.subplot(1,3,2)
    threshold = np.linspace(0.001,0.1,100)
    type1 = np.zeros(threshold.shape)
    for i,th in enumerate(threshold):
        type1[i] = (p<th).sum()/niter
    plt.plot(threshold,type1,'b.')
    plt.plot(threshold,threshold,'k')
    plt.ylabel('Number of Type 1 errors')
    plt.xlabel('Significance threshold')

    ax = plt.subplot(1,3,3)
    ss.probplot(t, dist=ss.t, sparams=(nsubj-1,), plot=ax,fit=False)
    lims = ax.get_xlim()
    plt.plot(lims,lims,'k:')
    pass 

def test_mmd2(s1=1,s2=1,nsubj=20,niter=1000,P =9):
    """Test for type I error rate in a case where the 
    variability of X1 and X2 is different. 
    """
    m = np.random.normal(0,1,(P,1))
    d = np.zeros((niter,))
    for i in range(niter):
        X1 = np.random.normal(0,s1,(nsubj,) + m.shape) + m
        X2 = np.random.normal(0,s2,(nsubj,) + m.shape) + m
        d[i] = rsa.mmd_unbiased_paired(X1,X2)
    sb.histplot(d)
    plt.axvline(d.mean(),color='k')

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



if __name__=='__main__':
    # individual_analysis()
    # cereb_cortical_rsa()
    test_test_sim_differences_type1()
    # test_mmd2()