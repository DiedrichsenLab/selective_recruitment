#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapwise selection recruitement analysis for the force speed and working memory tasks 
"""
# import packages
import numpy as np
from pathlib import Path
import os
import sys
# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.util as futil
import Functional_Fusion.matrix as fmatrix
import selective_recruitment.globals as gl
import selective_recruitment.recruite_ana as ra
import cortico_cereb_connectivity.globals as ccc_gl
import PcmPy as pcm
import SUITPy as suit
# import surfAnalysisPy.stats as sas
import surfAnalysisPy as sa
import matplotlib.pyplot as plt 

import nibabel as nb
from nilearn import plotting
import nitools as nt
import deepdish as dd
import scipy.stats as ss

import warnings

def load_data(ses_id = 'ses-02',
                subj = None,
                atlas_space='SUIT3',
                cortex = 'Icosahedron1002',
                type = "CondAll",
                mname = "MDTB_ses-s1_Icosahedron1002_L2Regression",
                reg = "A8",
                add_rest = False):
    """
    """    
    Y,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)
    X,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas="fs32k",
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)
    model_path = os.path.join(ccc_gl.conn_dir,atlas_space,'train',mname)
    fname = model_path + f"/{mname}_{reg}_avg.h5"
    json_name = model_path + f"/{mname}_{reg}_avg.json"
    conn_model = dd.io.load(fname)

    atlas,ainf = am.get_atlas('fs32k',gl.atlas_dir)
    label=[gl.atlas_dir+'/tpl-fs32k/'+cortex+'.L.label.gii',
           gl.atlas_dir+'/tpl-fs32k/'+cortex+'.R.label.gii']
    atlas.get_parcel(label,unite_struct=False)
    X, parcel_labels = ds.agg_parcels(X , 
                                         atlas.label_vector, 
                                         fcn=np.nanmean)
    YP = conn_model.predict(X)
    if add_rest:
        Y,_ = ra.add_rest_to_data(Y,info)
        YP,info = ra.add_rest_to_data(YP,info)

    return Y,YP,atlas,info

def calc_ttest_mean(res,c):
    """calculates the mean and t-test for a specific residual contrast
    """
    Cmean = c @ res
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        cmean = np.nanmean(Cmean,axis=0)
        std = np.nanstd(Cmean,axis=0)
        std[std==0]=np.nan
        N = np.sum(~np.isnan(Cmean),axis=0)
        T = cmean/std*np.sqrt(N)
    return cmean,T

def plot_data_flat(data,atlas_cereb,**kwargs):
    """plots the data on the flatmap
    """
    X = atlas_cereb.data_to_nifti(data)
    sdata = suit.flatmap.vol_to_surf(X)
    fig = suit.flatmap.plot(sdata,render='plotly',**kwargs)
    fig.show()

def check_roi_wise():
    atlas_space='SUIT3'
    Y,YP,cortex_atlas,info = load_data(ses_id = 'ses-01',
                                        subj = None,
                                        atlas_space='SUIT3',
                                        cortex = 'Icosahedron1002',
                                        type = "CondAll",
                                        mname = "MDTB_ses-s1_Icosahedron1002_L2Regression",
                                        reg = "A8",
                                        add_rest = True)
    # clean up the name string
    info["cond_name"] = info["cond_name"].str.rstrip("   ")

    res_c,coef_c,R2_c = ra.map_regress(YP,Y,fit_intercept=True,fit='common')

    # make a dataframe
    n_subj, n_cond, n_reg = res_parcel.shape
    DD = []
    for s in range(n_subj):
        for r in range(n_reg):
            d = info.copy()
            a = res_parcel[s, :, r]
            d["roi"] = (r+1)* np.ones([n_cond])
            d["res"] = a
            print(a)
            d["X"] = YP_parcel[s, :, r]
            d["Y"] = Y_parcel[s, :, r]
            d["sn"] = s
            d["intercept"] = coef_c[s, 0]
            d["slope"] = coef_c[s, 1]
            DD.append(d)

    DF = pd.concat(DD, ignore_index = True)

    # get roi_number from D
    roi_num = 19 #for M3R

    print(AnovaRM(data=DF[DF.cond_name != 'rest'][DF.roi == roi_num], depvar='res',
                subject='sn', within=['cond_name'], aggregate_func=np.mean).fit())
    return
if __name__=="__main__":
    atlas_space='SUIT3'
    Y,YP,cortex_atlas,info = load_data(ses_id = 'ses-02',
                                       atlas_space=atlas_space)

    # do regression and get residuals
    res_c,coef_c,R2_c = ra.map_regress(YP,Y,fit_intercept=True,fit='common')
    labels = gl.atlas_dir + f'/tpl-SUIT/atl-NettekovenSym68c32_space-SUIT_dseg.nii'
    res_parcel, ainfo, parcel_labels = ra.agg_data(res_c, "SUIT3", label = labels, unite_struct = False)
    Y_parcel, ainfo, parcel_labels = ra.agg_data(Y, "SUIT3", label = labels, unite_struct = False)
    YP_parcel, ainfo, parcel_labels = ra.agg_data(YP, "SUIT3", label = labels, unite_struct = False) 
    

    # Mapwise regression 
    res,coef,comvar = ra.map_pca(YP,Y,zero_mean=True,fit='separate')
    # calculate the mean and t-test for a specific residual
    c_overall = np.ones(13,)/13
    index = np.where(info.cond_name=='L6B_encode ')[0][0]
    c_cond = np.zeros(13,)
    c_cond[index] = 1
    c_task = np.ones(13,)/12
    c_task[-1]=0

    # Overall mean 
    mean_overall,T_overall = calc_ttest_mean(res,c_cond-c_task)
    plot_data_flat(T_overall,atlas_cereb)


    pass