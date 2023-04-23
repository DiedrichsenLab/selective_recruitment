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
import selective_recruitment.recruite_ana as ra
import selective_recruitment.globals as gl
import selective_recruitment.scripts.script_prep_sc as ss
import selective_recruitment.scripts.script_wm_sanity_checks as sa

import Functional_Fusion.dataset as fdata
import Functional_Fusion.atlas_map as am
from statsmodels.stats.anova import AnovaRM  # perform F test
import warnings
warnings.filterwarnings('ignore')


wkdir = 'A:\data\Cerebellum\CerebellumWorkingMemory\selective_recruit'
if not Path(wkdir).exists():
    wkdir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'
if not Path(wkdir).exists():
    wkdir = '/Users/jdiedrichsen/Data/wm_cerebellum/selective_recruit'

label_dict = {1: 'Enc2F', 2: 'Ret2F',
              3: 'Enc2B', 4: 'Ret2B',
              5: 'Enc4F', 6: 'Ret4F',
              7: 'Enc4B', 8: 'Ret4B',
              9: 'Enc6F', 10: 'Ret6F',
              11: 'Enc6B', 12: 'Ret6B',
              13: 'rest'}
marker_dict = {1: 'o', 2: 'X',
               3: 'o', 4: 'X',
               5: 'o', 6: 'X',
               7: 'o', 8: 'X',
               9: 'o', 10: 'X',
               11: 'o', 12: 'X',
               13: 's'}
color_dict = {1: 'b', 2: 'b',
              3: 'r', 4: 'r',
              5: 'b', 6: 'b',
              7: 'r', 8: 'r',
              9: 'b', 10: 'b',
              11: 'r', 12: 'r',
              13: 'g'}


def norm_within_category(df, category=['roi_name','sn'], value='Y', norm='zscore'):
    """
    Normalize the data within a category
    """
    gb = df.groupby(category)[value]
    if norm=='z-score':
        df[value + '_norm'] = gb.transform(lambda x: (x - x.mean()) / x.std())
    if norm=='norm':
        df[value + '_norm'] = gb.transform(lambda x: (x/np.sqrt((x**2).sum())))
    if norm=='mean':
        df[value + '_norm'] = gb.transform(lambda x: (x - x.mean()))
    if norm=='rmean':
        df[value + '_norm'] = gb.transform(lambda x: (x/x.mean()))
    return df

def plot_roi_differences():
    df_path = os.path.join(wkdir, "ROI_NettekovenSym68c32_conn_reg.tsv")
    dd = pd.read_csv(df_path, sep="\t")
    # print anova results
    names = dd.roi_name.values.astype(str)

    # add a new column determining side (hemisphere)

    dd["roi_super"] = dd["roi_name"].str[0]
    dd["roi_sub"] = dd["roi_name"].str[1]
    dd["side"] = dd["roi_name"].str[2]
    D=dd.loc[(dd.cond_name != "rest") & (dd.roi_super=='D')]

    # Make sn column into an integer
    D['sn']=D['sn'].astype(int)
    D['cond_num']=D.phase*6+D.recall*3+D.load/2
    D = norm_within_category(D, category=['roi_name'], value='Y', norm='mean')
    plt.figure()
    d1 = (1,0)
    d2 = (3,3)
    ax = sns.lineplot(data=D, x = 'cond_num', y = 'Y_norm', hue = 'roi_name',style='roi_name',
                palette=['r','b','g','k','r','b','g','k'],
                dashes=[d1,d1,d1,d1,d2,d2,d2,d2],
                err_style=None)
    plt.xticks(rotation = 90)
    """
    plt.subplot(1,2,1)
    sns.lineplot(data=D.loc[D.roi_name=='D3R'],x='cond_name',y='Y',hue='sn',err_style=None)
    plt.xticks(rotation = 90)
    plt.subplot(1,2,2)
    sns.lineplot(data=D.loc[D.roi_name=='D3R'],x='cond_name',y='Y_norm',hue='sn',err_style=None)

    plt.xticks(rotation = 90)
    """
    return D

if __name__ == "__main__":
    D = plot_roi_differences()
    anov = AnovaRM(data=D, depvar='Y_norm',
                  subject='sn',
                  within= ["cond_name", "roi_name"],
                  aggregate_func=np.mean).fit()
    print(anov)
    D = norm_within_category(D, category=['roi_name','sn'], value='Y_norm', norm='mean')
    A = pd.pivot_table(data=D,index='roi_name',columns='cond_name',values='Y',aggfunc=np.mean)
    C=A.values
    C=C/np.sqrt((C**2).sum(axis=1,keepdims=True))
    B = C@C.T
    plt.figure()
    plt.imshow(B)
    ax = plt.gca()
    ax.set_yticklabels(A.index)


    W,V = plotting.calc_mds(A.values)
    plotting.plot_MDS(W[:,0],W[:,1],A.index)
    pass
