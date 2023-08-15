#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to make label files for the cortical and cerebellar rois

Created on 07/03/2023
Author: Ladan Shahshahani
"""
# import packages
import os
import numpy as np
import pandas as pd
import nitools as nt
import nibabel as nb
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from pathlib import Path
from SUITPy import flatmap
import PcmPy as pcm

import selective_recruitment.plotting as plotting
import regress as ra
import selective_recruitment.globals as gl
import selective_recruitment.region as sroi
import selective_recruitment.data as data
import selective_recruitment.regress as reg

import Functional_Fusion.atlas_map as am

from statsmodels.stats.anova import AnovaRM  # perform F test


# setting working directory
wkdir = '/Volumes/diedrichsen_data$/data/Cerebellum/Demand/selective_recruit'
if not Path(wkdir).exists():
    wkdir = 'A:\data\Cerebellum\Demand\selective_recruit'
if not Path(wkdir).exists():
    wkdir = '/srv/diedrichsen/data/Cerebellum/Demand/selective_recruit'

def demand_Davrg():
    D = data.get_summary_conn(dataset = "Demand",
                 ses_id = 'ses-01',
                 type = "CondHalf",
                 cerebellum_roi ='NettekovenSym32',
                 cerebellum_roi_selected = ['D..'],
                 cortex_roi = "Icosahedron1002",
                 add_rest = True,
                 mname_base = 'MDTB_all_Icosahedron1002_L2regression',
                 mname_ext = '_a8',
                 crossed = True)
    D = reg.roi_regress(D,fit_intercept=True)
    D.to_csv(wkdir + '/ROI_D.avrg_cMDTB.tsv',sep='\t')

if __name__ == "__main__":
    demand_Davrg()
    # create an instance of the dataset class
    # D = get_summary_roi(dataset = "MDTB",
    #              ses_id = 'ses-s1',
    #              type = "CondAll",
    #              subj=['sub-02'],
    #              cerebellum_roi = None, # "NettekovenSym32",
    #              cerebellum_roi_selected= ['D1L','D2L'],
    #              cortex_roi = None, # "Icosahedron1002",
    #              cortex_roi_selected=[2,3],
    #              add_rest = True)
    # pass
    # pass
    # df = get_summary_conn(dataset = "WMFS",
    #                  ses_id = 'ses-02',
    #                  type = "CondAll",
    #                  subj = [1,2],
    #                  atlas_space = "SUIT3",
    #                  cerebellum_roi = "NettekovenSym32",
    #                  cortex_roi = "Icosahedron1002",
    #                 add_rest = True,
    #                  mname_base = "MDTB_all_Icosahedron1002_L2Regression",
    #                  mname_ext = "_A8",
    #                  crossed = True)
    # pass
    # pass
    # test case
    # D = get_summary_conn(dataset = "WMFS",
    #                  ses_id = 'ses-02',
    #                  subj = None,
    #                  atlas_space = "SUIT3",
    #                  cerebellum_roi = "NettekovenSym68c32",
    #                  cortex_roi = "Icosahedron1002",
    #                  type = "CondHalf",
    #                  add_rest = True,
    #                  mname_base = "MDTB_ses-s1",
    #                  mmethod = "L2Regression_A8",
    #                  crossed = True)
    pass