#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to test the pca function
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
import selective_recruitment.data as data
import selective_recruitment.regress as reg

from statsmodels.stats.anova import AnovaRM  # perform F test

def test_pca():
    z=np.random.normal(0,1,(20,))
    x=z + np.random.normal(0,1,(20,))+10
    y=3* z + np.random.normal(0,1,(20,))-10
    vec = np.ones((20,),dtype=int)
    D = pd.DataFrame({'sn':vec,'roi':vec,'X':x,'Y':y})
    D = reg.roi_pca(D,zero_mean=True)
    sns.scatterplot(x='X',y='Y',data=D)
    xrange = np.array([D.X.min(),D.X.max()])
    ypred = xrange*D.slope.mean()+D.intercept.mean()
    plt.plot(xrange,ypred,'k-')

    pass



if __name__ == "__main__":
    test_pca()
    pass
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