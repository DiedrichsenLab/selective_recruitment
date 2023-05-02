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


def load_data(ses_id = 'ses-02',
                subj = None,
                atlas_space='SUIT3',
                cortex = 'Icosahedron1002',
                type = "CondAll",
                mname = "MDTB_all_Icosahedron1002_L2Regression",
                reg = "A8",):
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
    Y,_ = ra.add_rest_to_data(Y,info)
    YP,info = ra.add_rest_to_data(YP,info)

    return Y,YP,atlas,info

if __name__=="__main__":
    atlas_space='SUIT3'
    Y,YP,cortex_atlas,info = load_data(ses_id = 'ses-02',
                                       atlas_space=atlas_space)
    res,coef,R2 = ra.map_regress(Y,YP,fit_intercept=True,fit='common')
    atlas_cereb,ainf = am.get_atlas(atlas_space,gl.atlas_dir)
    
    # calculate the mean and t-test for a specific residual
    index = np.where(info.cond_name=='L2F_encode ')[0][0]
    mean = np.nanmean(res[:,index,:],axis=0)
    std = np.nanstd(res[:,index,:],axis=0)
    N = np.sum(~np.isnan(res[:,index,:]),axis=0)
    T = mean/std*np.sqrt(N)
    data = np.nanmean(Y[:,index,:],axis=0)

    X = atlas_cereb.data_to_nifti(T)
    sdata = suit.flatmap.vol_to_surf(X)
    fig = suit.flatmap.plot(sdata,render='plotly')
    fig.show()
    pass