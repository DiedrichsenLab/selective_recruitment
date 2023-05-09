"""
script to prepare dataframe for scatterplots
@ Ladan Shahshahani Joern Diedrichsen Feb 19 2023
"""
from pathlib import Path
import pandas as pd
from collections import defaultdict
import deepdish as dd
import numpy as np

import nibabel as nb
import Functional_Fusion.dataset as fdata 
import Functional_Fusion.atlas_map as am
import regress as ra
import selective_recruitment.globals as gl
import selective_recruitment.region as sroi

out_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'
if not Path(out_dir).exists():
    out_dir = '/srv/diedrichsen/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'

def get_summary_data(dataset = "WMFS", 
                     ses_id = 'ses-02', 
                     atlas_space = "fs32k",
                     atlas_roi = "glasser",
                     type = "CondHalf", 
                     unite_struct = False,
                     add_rest = True):
    """
    """
    # get data
    tensor, info, _ = fdata.get_dataset(gl.base_dir,
                                        dataset,atlas= atlas_space,
                                        sess=ses_id,
                                        type=type, 
                                        info_only=False)

    # get label file
    if atlas_space == "fs32k":
        labels = []
        for hemi in ['L', 'R']:
            labels.append(gl.atlas_dir + f'/tpl-fs32k/{atlas_roi}.{hemi}.label.gii')
        var = "X" # will be used when saving the dataframe

    else:
        labels = gl.atlas_dir + f'/tpl-SUIT/{atlas_roi}_dseg.nii'
        var = "Y"

    # get average data per parcel
    parcel_data, ainfo, parcel_labels = ra.agg_data(tensor, atlas_space, labels, unite_struct = unite_struct)
    
    # use lookuptable to get region info
    region_info = sroi.get_label_names(atlas_roi, atlas_space= atlas_space) 
    
    # add rest condition for control?
    if add_rest:
        parcel_data,info = ra.add_rest_to_data(parcel_data,info)

    # Transform into a dataframe with Yhat and Y data 
    n_subj,n_cond,n_roi = parcel_data.shape

    summary_list = [] 
    for i in range(n_subj):
        for r in range(n_roi):
            info_sub = info.copy()
            vec = np.ones((len(info_sub),))
            info_sub["sn"]    = i * vec
            info_sub["roi"]   = parcel_labels[r] * vec
            info_sub["roi_name"] = region_info[r+1]
            info_sub[var]     = parcel_data[i,:,r]

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0,ignore_index=True)

    return summary_df


if __name__ == "__main__":
    D = get_summary_conn(dataset="WMFS",
                        ses_id='ses-02',
                        type="CondHalf",
                        cerebellum_roi='NettekovenSym68c32integLR',
                        cortex_roi="Icosahedron1002",
                        add_rest=True)
    pass


    




