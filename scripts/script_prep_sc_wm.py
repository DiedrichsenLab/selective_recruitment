"""
script to prepare dataframe for scatterplots
@ Ladan Shahshahani Jan 30 2023 12:57
"""
import os
import numpy as np
import deepdish as dd
import pathlib as Path
import pandas as pd
import re
import sys
from collections import defaultdict
sys.path.append('../cortico-cereb_connectivity')
sys.path.append('../selective_recruitment')
sys.path.append('..')
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import prepare_data as prep
import select_recruite as sr


# run this function to make sure that you have saved data tensors
def prep_tensor():
   prep.save_data_tensor(dataset = "WMFS",
                         atlas='SUIT3',
                         ses_id='ses-02',
                         type="CondHalf")
   prep.save_data_tensor(dataset = "WMFS",
                         atlas='fs32k',
                         ses_id='ses-02',
                         type="CondHalf")
   return

#  
def prep_sc_whole(outpath = prep.conn_dir+ 'WMFS/'):
    """
    getting the summary dataframe for the scatterplot with the averaged over structures
    """
    df = sr.get_summary_roi(outpath,
                        dataset_name = "WMFS", 
                        agg_whole=True, 
                        save_tensor=False)
    # save the dataframe for later
    filepath = os.path.join(outpath, 'sc_df_whole_ses-02.tsv')
    df.to_csv(filepath, index = False, sep='\t')
    return

# 
def prep_sc_roi(outpath = prep.conn_dir + 'WMFS/'):
    """
    Getting the summary dataframe for the scatterplot in an ROI-wise manner
    """
    df = sr.get_summary_roi(outpath,
                        dataset_name = "WMFS", 
                        agg_whole=False, 
                        cerebellum="Verbal2Back", 
                        cortex="Verbal2Back.32k", 
                        save_tensor = False)

    # save the dataframe for later
    filepath = os.path.join(outpath, 'sc_df_VWM_ses-02.tsv')
    df.to_csv(filepath, index = False, sep='\t')
    return

# 
def prep_sc_conn(outpath = prep.conn_dir + 'WMFS/', 
                 conn_dataset = "MDTB", 
                 conn_ses_id = "ses-s1", 
                 log_alpha = 8):
    """
    Getting the summary dataframe for the scatterplot in an ROI-wise manner
    """
    df = sr.get_summary_conn(outpath, 
                             dataset_name = "WMFS",
                             method = 'L2Regression',
                             cereb_roi = "Verbal2Back", 
                             parcellation = 'Icosahedron-1002_Sym.32k', 
                             conn_dataset = conn_dataset,
                             conn_ses_id  = conn_ses_id,
                             log_alpha = log_alpha, 
                             ses_id = 'ses-02',
                             type = "CondHalf", 
                             save_tensor = False)

    # save the dataframe for later
    filepath = os.path.join(outpath, f'sc_df_VWM_conn_{conn_dataset}_ses-02.tsv')
    df.to_csv(filepath, index = False, sep='\t')
    return


if __name__ == "__main__":
    # prep_tensor()
    prep_sc_whole()
    prep_sc_roi()
    prep_sc_conn(conn_dataset = "Demand", 
                 conn_ses_id = "ses-01", 
                 log_alpha = 8)

    prep_sc_conn(conn_dataset = "Nishimoto", 
                 conn_ses_id = "ses-01", 
                 log_alpha = 8)