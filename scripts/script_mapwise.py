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
    X,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)
    Y,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas="fs32k",
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)
    model_path = os.path.join(ccc_gl.conn_dir,atlas_space,'train',mname)
    fname = model_path + f"/{m}_{reg}_avg.h5"
    json_name = model_path + f"/{m}_{reg}_{sub}.json"
    conn_model = dd.io.load(fname)

    atlas = am.get_atlas('fs32k',futil.atlas_dir)
    label=[cortex+'.L.label.gii',cortex+'.R.label.gii']
    atlas.get_parcel(label,unite_struct=False)
    X, parcel_labels = ds.agg_parcels(X , 
                                         atlas.label_vector, 
                                         fcn=np.nanmean)



    return X,Y


if __name__=="__main__":
    X,Y = load_data(ses_id = 'ses-02')
    pass