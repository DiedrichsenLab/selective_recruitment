#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to make label files for the cortical and cerebellar rois

Author: Ladan Shahshahani, Joern Diedrichsen
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

from statsmodels.stats.anova import AnovaRM  # perform F test
# import warnings
# warnings.filterwarnings('ignore')


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


def roi_difference(df,
             xvar = "cond_name",
             hue = "roi_name",
             depvar = "Y",
             sub_roi = None,
             roi = "D",
             var = ["cond_name", "roi_name"]):
    """ roi_difference plots and tests for differences between rois
    """
    # get D regions alone?
    names = df.roi_name.values.astype(str)
    mask_roi = np.char.startswith(names, roi)

    # add a new column determining side (hemisphere)
    df["side"] = df["roi_name"].str[2]

    # add a column determining anterior posterior
    mask_anteriors = np.char.endswith(names, "A")
    df["AP"] = ""
    df["AP"].loc[mask_anteriors] = "A"
    df["AP"].loc[np.logical_not(mask_anteriors)] = "P"

    # add a new column that defines the index assigned to the region
    # for D2 it will be 2, for D3 it will be 3
    df["sub_roi_index"] = df["roi_name"].str[1]

    # add a new column determining side (hemisphere)
    df["side"] = df["roi_name"].str[2]

    # get Ds
    DD_D = df.loc[(mask_roi)]
    # get the specific region
    if sub_roi is not None:
        DD_D = DD_D.loc[DD_D.sub_roi_index == sub_roi]

    # barplots
    plt.figure()
    ax = sns.lineplot(data=DD_D.loc[(df.cond_name != "rest")], x = xvar, y = depvar,
                    errwidth=0.5, hue = hue)
    plt.xticks(rotation = 90)
    anov = AnovaRM(data=df, depvar=depvar,
                  subject='sn', within=var, aggregate_func=np.mean).fit()
    return anov

def norm_within_category(df, category=['roi_name','sn'], value='Y', norm='z-score'):
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
    df_path = os.path.join(wkdir, "wm_ROI_NettekovenSym68c32AP_conn_reg.tsv")
    D = pd.read_csv(df_path, sep="\t")
    D, cond_map = prep_roi_comparison(D)
    D = norm_within_category(D, category=["sn", "roi_name"], value='Y', norm="mean")
    D = D.loc[D.roi != 0]
    D = D.loc[D.cond_name != "rest"]
    D["AP"] = D["roi_name"].str[4]
    # D = plot_roi_differences(D,cond_map)
    print(AnovaRM(data=D[(D.cond_name != 'rest')],
                  depvar='Y_norm',
                  subject='sn',
                  within=['phase'],
                  aggregate_func=np.mean).fit())
    # anov = AnovaRM(data=D, depvar='Y_norm',
    #               subject='sn',
    #               within= ["cond_name", "roi_name"],
    #               aggregate_func=np.mean).fit()
    # print(anov)


    # plt.figure()
    # D = norm_within_category(D, category=['roi_name','sn'], value='Y_norm', norm='mean')
    # A = pd.pivot_table(data=D,index='roi_name',columns='cond_name',values='Y_norm',aggfunc=np.mean)
    # C=A.values
    # C=C/np.sqrt((C**2).sum(axis=1,keepdims=True))
    # B = C@C.T

    # K=3
    # W,V = plotting.calc_mds(A.values,K=K)
    # # phase, load, and recall
    # vs = np.array([[-1, 1,-1, 1,-1,1,-1,1,-1,1,-1,1],
    #               [-1,-1,-1,-1, 0,0, 0,0, 1,1, 1,1],
    #               [1,1, -1, -1, 1, 1, -1, -1,1,1, -1, -1]])
    # vs = vs/np.sqrt((vs**2).sum(axis=1,keepdims=True))
    # proj_vs = V @ vs.T
    # red =(0.8,0.2,0.2)
    # gray = (0.5,0.5,0.5)
    # lb = (0.2,0.5,1.0)
    # db = (0.0,0.1,0.6)
    # pal = [red,red,gray,gray,lb,lb,db,db]

    # if K==2:
    #     plotting.plot_mds(W[:,0],W[:,1],A.index,
    #                       colors=pal,
    #                       vectors=proj_vs,
    #                       v_labels = ['retrieval','load+','backwards'])
    # elif K==3:
    #     plotting.plot_mds3(W[:,0],W[:,1],W[:,2],A.index,
    #                         colors=pal,
    #                         vectors=proj_vs,
    #                         v_labels = ['retrieval','load+','backwards'])
    pass

# write a function that takes a dataframe and a list of columns and returns a dataframe with the mean of the columns
# for each subject
# def mean_within_subject(df, columns):
#     df = df.groupby(columns).mean()
#     return df
