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
import pickle
import generativeMRF

from pathlib import Path
from SUITPy import flatmap
import PcmPy as pcm

import selective_recruitment.plotting as plotting
import regress as ra
import selective_recruitment.globals as gl
import selective_recruitment.scripts.script_prep_sc as ss

import Functional_Fusion.dataset as fdata
import Functional_Fusion.atlas_map as am
from statsmodels.stats.anova import AnovaRM  # perform F test
# import warnings
# warnings.filterwarnings('ignore')

# TODO: roi differences between cortical rois using glasser/power etc parcellations


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

def prep_roi_comparison(dd):
    dd["roi_super"] = dd["roi_name"].str[0]
    dd["roi_sub"] = dd["roi_name"].str[1]
    dd["side"] = dd["roi_name"].str[2]
    D=dd.loc[(dd.cond_name != "rest") & (dd.roi_super=='D')]
    D['sn']=D['sn'].astype(int)
    D['cond_num']=D.phase*6+(1-D.recall)*3+D.load/2
    cond_map = D[['cond_num','cond_name']].drop_duplicates()
    cond_map.sort_values(by='cond_num', inplace=True)
    return D, cond_map

def plot_roi_differences(D, cond_map, depvar = "Y_norm", var = ["cond_name", "roi_name"]):
    # print anova results
    

    # Make sn column into an integer
    D = norm_within_category(D, category=['roi_name','sn'], value=depvar[0], norm='mean')

    anov = AnovaRM(data=D, depvar=depvar,
                  subject='sn', within=var, aggregate_func=np.mean).fit()
    print(anov)
    plt.figure()
    # Define styles and colors
    d1 = (1,0)
    d2 = (3,3)
    red =(0.8,0.2,0.2)
    gray = (0.5,0.5,0.5)
    lb = (0.2,0.5,1.0)
    db = (0.0,0.1,0.6)
    ax = sns.lineplot(data=D, x = 'cond_num', y = depvar, hue = 'roi_name',style='roi_name',
                palette=[red,gray,lb,db,red,gray,lb,db],
                dashes=[d1,d1,d1,d1,d2,d2,d2,d2],
                err_style=None)
    # Find mapping between cond_name and cond_num
    ax.set_xticks(np.arange(12)+1)
    ax.set_xticklabels(cond_map.cond_name.values, rotation=45)
    return D

if __name__ == "__main__":
    df_path = os.path.join(wkdir, "ROI_NettekovenSym68c32_conn_reg.tsv")
    D = pd.read_csv(df_path, sep="\t")
    D, cond_map = prep_roi_comparison(D)
    D = plot_roi_differences(D,cond_map)
    anov = AnovaRM(data=D, depvar='Y_norm',
                  subject='sn',
                  within= ["cond_name", "roi_name"],
                  aggregate_func=np.mean).fit()
    print(anov)


    plt.figure()
    D = norm_within_category(D, category=['roi_name','sn'], value='Y_norm', norm='mean')
    A = pd.pivot_table(data=D,index='roi_name',columns='cond_name',values='Y_norm',aggfunc=np.mean)
    C=A.values
    C=C/np.sqrt((C**2).sum(axis=1,keepdims=True))
    B = C@C.T

    K=3
    W,V = plotting.calc_mds(A.values,K=K)
    # phase, load, and recall
    vs = np.array([[-1, 1,-1, 1,-1,1,-1,1,-1,1,-1,1],
                  [-1,-1,-1,-1, 0,0, 0,0, 1,1, 1,1],
                  [1,1, -1, -1, 1, 1, -1, -1,1,1, -1, -1]])
    vs = vs/np.sqrt((vs**2).sum(axis=1,keepdims=True))
    proj_vs = V @ vs.T
    red =(0.8,0.2,0.2)
    gray = (0.5,0.5,0.5)
    lb = (0.2,0.5,1.0)
    db = (0.0,0.1,0.6)
    pal = [red,red,gray,gray,lb,lb,db,db]

    if K==2:
        plotting.plot_mds(W[:,0],W[:,1],A.index,
                          colors=pal,
                          vectors=proj_vs,
                          v_labels = ['retrieval','load+','backwards'])
    elif K==3:
        plotting.plot_mds3(W[:,0],W[:,1],W[:,2],A.index,
                            colors=pal,
                            vectors=proj_vs,
                            v_labels = ['retrieval','load+','backwards'])
    pass
