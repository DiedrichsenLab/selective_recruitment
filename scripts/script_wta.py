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
import selective_recruitment.region as sroi

import cortico_cereb_connectivity.model as model

import Functional_Fusion.dataset as fdata
import Functional_Fusion.atlas_map as am

from statsmodels.stats.anova import AnovaRM  # perform F test

def calcX2(D,rows='cereb_roi',cols='assigned_num'):
    observed = pd.crosstab(D[rows],D[cols], margins=True)
    expected = np.outer(observed["All"][0:-1], observed.loc["All"][0:-1]) /observed['All']['All']
    observed = pd.crosstab(D[rows],D[cols],margins=False)
    return sum(sum((observed.values-expected)**2/expected))

def randomize_column(df,colname):
    df_copy = df.copy()
    var = df_copy[colname].values
    np.random.shuffle(var)
    df_copy[colname]=var
    return df_copy

def randomization_test(D,shuffle, rows,numIterations=500,sides=2, nbins = 10):    
    listOfTS =  np.array(range(numIterations),dtype = 'float64')
    for i in range(numIterations):
        #1. Randomly shuffle the data 
        S= randomize_column(D,shuffle)         
        #2. Calculate test statistics 
        listOfTS[i] = calcX2(S, rows=rows,cols=shuffle)

    # 3. Calculate the real test statistic 
    realTS = calcX2(D, rows=rows,cols=shuffle)

    # 4. Plot a histogram of the 
    plt.hist(listOfTS,bins= nbins)
    plt.axvline(x=realTS, color='k')
    
    # 5. determine p-value (one sided) with rejection region above the threshold 
    if sides==1: 
        p=sum(listOfTS>=realTS)/len(listOfTS)
    # 5. determine p-value (two sided) with rejection region outside the threshold 
    
    
    if sides==2:
        plt.axvline(x=-realTS, color='r')
        p=sum(np.absolute(listOfTS)>=np.absolute(realTS))/len(listOfTS)
    print(' P-value of the randomisation test is p= ',p)
    return p

def assign_winner(cortex_roi = "glasser",
                  dataset = "WMFS",  
                  ses_id = "ses-02", 
                  type = "CondAll", 
                  roi_sup = "D", 
                  cerebellum_roi = "NettekovenSym68c32",
                  atlas_cerebellum = "SUIT3", ):
    """
    """

    # get label files 
    cortex_label = []
    for hemi in ['L', 'R']:
        cortex_label.append(gl.atlas_dir + '/tpl-fs32k' + f'/{cortex_roi}.{hemi}.label.gii')
    region_info_fs = sroi.get_label_names(cortex_roi, atlas_space = "fs32k")


    cerebellum_label = gl.atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum_roi}_space-SUIT_dseg.nii'
    # use lookuptable to get region info
    region_info_cereb = sroi.get_label_names(cerebellum_roi)[1:] 
    reg_names = [i for i in region_info_cereb if roi_sup in i]
    reg_nums = [region_info_cereb.index(i) for i in region_info_cereb if roi_sup in i]


    # get data tensors for the cerebellum and cortex
    t_cereb, info, _ = fdata.get_dataset(gl.base_dir,dataset,atlas=atlas_cerebellum,sess=ses_id,type=type, info_only=False)
    t_fs, info, _ = fdata.get_dataset(gl.base_dir,dataset,atlas="fs32k",sess=ses_id,type=type, info_only=False)

    # get parcels average
    ## corticals:
    X_parcel, ainfo, X_parcel_labels = ra.agg_data(t_fs, "fs32k", cortex_label, unite_struct = True)
    ## cerebellars:
    Y_parcel, ainfo, Y_parcel_labels = ra.agg_data(t_cereb, atlas_cerebellum, cerebellum_label)
    # get the subset of regions
    Y_parcel = Y_parcel[:, :, reg_nums]

    # get group to select cortical regions
    # Xgroup = np.mean(np.mean(X_parcel, axis = 0), axis = 0)
    # Q = np.percentile(Xgroup, 90)
    # thresholded = np.argwhere(Xgroup>Q)
    # # get the data for the selected regions
    # XX = X_parcel[:, :, thresholded[:, 0]]
    XX = X_parcel.copy()

    # TODO: make a label file for the selected regions


    # get the names and numbers of the selected regions
    # reg_names_fs = [region_info_fs[int(i)] for i in thresholded+1]
    reg_names_fs = region_info_fs.copy()

    # use WTA model from connectivity module to assign corticals to each cerebellar roi
    ## create an instance of the WTA model
    WTA = model.WTA()

    # prep summary
    D = []

    ## loop over subjects 
    n_subj, n_cond, n_parcel_cereb = Y_parcel.shape
    n_parcel_fs = XX.shape[2]
    ### initialize weight and labels arrays
    weight_ = np.zeros([n_subj, n_parcel_cereb, n_parcel_fs])
    assigned_label_ = np.zeros([n_subj, n_parcel_cereb])
    for s in range(n_subj):
        # threshold

        ## get coef and labels by fitting the model
        weight_[s], assigned_label_[s] = WTA.fit(XX[s], Y_parcel[s])
        # print(assigned_label_[s])
        # print([reg_names_fs[int(i)] for i in assigned_label_[s]])

        dd = pd.DataFrame()
        dd["cereb_roi"] = reg_names
        dd["assigned_num"] = assigned_label_[s].astype(int)
        # dd["assigned_name"] = [reg_names_fs[int(i)] for i in assigned_label_[s]]
        dd["sn"] = f"sub-{s+1:02d}"

        D.append(dd)



    return pd.concat(D, ignore_index=True)


if __name__ == "__main__":
    df = assign_winner(cortex_roi = "glasser", #Icosahedron42
                  dataset = "Demand",  
                  ses_id = "ses-01", 
                  type = "CondAll", 
                  roi_sup = "D", 
                  cerebellum_roi = "NettekovenSym68c32",
                  atlas_cerebellum = "SUIT3")

    # get D regions
    DD = df.loc[df.cereb_roi.str.contains("D")]

    p = randomization_test(DD,
                           rows= "cereb_roi", 
                           shuffle = "assigned_num",
                           numIterations=100,
                           sides=2, 
                           nbins = 100)

    print(p)
    
    