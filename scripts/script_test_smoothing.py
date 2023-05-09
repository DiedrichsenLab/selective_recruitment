# import packages
import enum
import numpy as np
import pandas as pd
from pathlib import Path
# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import selective_recruitment.globals as gl
import regress as ra
import cortico_cereb_connectivity.evaluation as ccev
import SUITPy as suit
import surfAnalysisPy as sa
import matplotlib.pyplot as plt 
from scipy.stats import norm, ttest_1samp
from nipype.interfaces.workbench import CiftiSmooth

#
import os
import subprocess
import nibabel as nb
from nilearn import plotting
import nitools as nt
import SUITPy.flatmap as flatmap
from nilearn import plotting
import scipy.stats as ss
wkdir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'
def test_cifti_smooth(subject = "group", surf_sigma = 2, vol_sigma = 2, direction = "COLUMN", prefix = "s"):
    
    # preparing the data and atlases for all the structures
    Data = ds.get_dataset_class(gl.base_dir, dataset="WMFS")
    surfs = [Data.atlas_dir + f'/tpl-fs32k/tpl_fs32k_hemi-{h}_inflated.surf.gii' for i, h in enumerate(['L', 'R'])]
    fpath = f"{wkdir}/data/{subject}"
    cifti_name = f"load_recall_space_fs32k_CondAll_{subject}.dscalar.nii"
    cifti_file = f"{fpath}/{cifti_name}"
    # make up the command
    smooth_cmd = f"wb_command -cifti-smoothing {cifti_file} {surf_sigma} {vol_sigma} {direction} {fpath}/{prefix}{cifti_name} -left-surface {surfs[0]} -right-surface {surfs[1]}"
    subprocess.run(smooth_cmd, shell=True)
    
    return 

if __name__ == "__main__":
    B = test_cifti_smooth()