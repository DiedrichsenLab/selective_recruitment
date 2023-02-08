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

# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import matplotlib.pyplot as plt
# modules from connectivity
import cortico_cereb_connectivity.prepare_data as cprep

import os
import nibabel as nb
import nitools as nt
import PcmPy as pcm
import selective_recruitment.rsa as rsa


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
    G1,Ginf,m1 = rsa.calc_rsa(DCereb,info,center=False,reorder=['phase','recall'])
    DCortex, info, dataset = ds.get_dataset(base_dir,
                                                'WMFS',
                                                atlas='fs32k',
                                                sess='ses-02',
                                                type='CondRun')
    G2,Ginf,m2 = rsa.calc_rsa(DCortex,info,center=False,reorder=['phase','recall'])
    
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.imshow(G1.mean(axis=0))
    plt.subplot(1,2,2)
    plt.imshow(G2.mean(axis=0))




if __name__=='__main__':
    cereb_cortical_rsa()