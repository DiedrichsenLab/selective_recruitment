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



if __name__=='__main__':
    individual_analysis()
    # cereb_cortical_rsa()