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
import data as cdata # from connectivity 
import select_recruite as sr


# run this function to make sure that you have saved data tensors
def prep_tensor(dataset_name = "WMFS", ses_id = 'ses-02'):
   cdata.save_data_tensor(dataset = dataset_name,
                         atlas='SUIT3',
                         ses_id=ses_id,
                         type="CondHalf")
   cdata.save_data_tensor(dataset = dataset_name,
                         atlas='fs32k',
                         ses_id=ses_id,
                         type="CondHalf")
   return

def prep_sc_whole(outpath = cdata.conn_dir, 
                dataset_name = "WMFS", 
                cerebellum = "SUIT3", 
                cortex = "tpl-fs32k_mask", 
                ses_id = 'ses-02'):
    """
    Getting the summary dataframe for the scatterplot in an ROI-wise manner
    """
    df = sr.get_summary(outpath,
                        dataset_name = dataset_name, 
                        ses_id =ses_id,
                        cerebellum=cerebellum, 
                        cortex=cortex, 
                        predict = False,
                        unite_struct=True,
                        save_tensor = False)

    # save the dataframe for later
    filepath = os.path.join(outpath, dataset_name, f'sc_{dataset_name}_{ses_id}_whole.tsv')
    df.to_csv(filepath, index = False, sep='\t')
    return

def prep_sc_roi(outpath = cdata.conn_dir, 
                dataset_name = "WMFS", 
                cerebellum = "Verbal2Back", 
                cortex = "Verbal2Back.32k", 
                ses_id = 'ses-02'):
    """
    Getting the summary dataframe for the scatterplot in an ROI-wise manner
    """
    df = sr.get_summary(outpath,
                        dataset_name = dataset_name, 
                        ses_id =ses_id,
                        cerebellum=cerebellum, 
                        cortex=cortex, 
                        predict = False,
                        unite_struct=True,
                        save_tensor = False)

    # save the dataframe for later
    filepath = os.path.join(outpath, dataset_name, f'sc_{dataset_name}_{ses_id}_{cerebellum}.tsv')
    df.to_csv(filepath, index = False, sep='\t')
    return

# 
def prep_sc_conn(outpath = cdata.conn_dir, 
                 dataset_name = "WMFS", 
                 conn_dataset = "MDTB", 
                 conn_ses_id = "ses-s1", 
                 ses_id = 'ses-02', 
                 cerebellum = "Verbal2Back", 
                 log_alpha = 8):
    """
    Getting the summary dataframe for the scatterplot in an ROI-wise manner
    """
    df = sr.get_summary(outpath = outpath, 
                        dataset_name = dataset_name,
                        conn_method="L2Regression", 
                        cerebellum = cerebellum, 
                        cortex = 'Icosahedron-1002_Sym.32k', 
                        conn_dataset = conn_dataset,
                        conn_ses_id  = conn_ses_id,
                        log_alpha = log_alpha, 
                        ses_id = ses_id,
                        type = "CondHalf",
                        predict = True, 
                        unite_struct=False,  
                        save_tensor = False)

    # save the dataframe for later
    filepath = os.path.join(outpath, dataset_name, f'sc_conn_{conn_dataset}_{dataset_name}_{ses_id}_{cerebellum}.tsv')
    df.to_csv(filepath, index = False, sep='\t')
    return


if __name__ == "__main__":
    # prep_tensor(dataset_name = "WMFS", ses_id = "ses-01")
    # prep_tensor(dataset_name = "WMFS", ses_id = "ses-01")
    """
    For WMFS
    """
    # prep_sc_whole(outpath = cdata.conn_dir, 
    #               dataset_name = "WMFS", 
    #               ses_id = 'ses-02')

    # prep_sc_whole(outpath = cdata.conn_dir, 
    #               dataset_name = "WMFS", 
    #               ses_id = 'ses-01')

    # prep_sc_roi(outpath = cdata.conn_dir, 
    #             dataset_name = "WMFS", 
    #             cerebellum = "Verbal2Back", 
    #             cortex = "Verbal2Back.32k", 
    #             ses_id = 'ses-02')

    # prep_sc_conn(outpath = cdata.conn_dir, 
    #              dataset_name = "WMFS", 
    #              conn_dataset = "MDTB", 
    #              conn_ses_id = "ses-s1", 
    #              ses_id = 'ses-02', 
    #              cerebellum = "Verbal2Back", 
    #              log_alpha = 8)

    # prep_sc_conn(outpath = cdata.conn_dir, 
    #              dataset_name = "WMFS", 
    #              conn_dataset = "MDTB", 
    #              conn_ses_id = "ses-s1", 
    #              ses_id = 'ses-01', 
    #              cerebellum = "MDTB10", 
    #              log_alpha = 8)

    """
    For Demand
    """
    # prep_sc_whole(outpath = cdata.conn_dir, 
    #               dataset_name = "Demand", 
    #               ses_id = 'ses-01')

    # prep_sc_roi(outpath = cdata.conn_dir, 
    #             dataset_name = "Demand", 
    #             cerebellum = "Verbal2Back", 
    #             cortex = "Verbal2Back.32k", 
    #             ses_id = 'ses-01')

    prep_sc_conn(outpath = cdata.conn_dir, 
                 dataset_name = "Demand", 
                 conn_dataset = "MDTB", 
                 conn_ses_id = "ses-s1", 
                 ses_id = 'ses-01', 
                 cerebellum = "Verbal2Back", 
                 log_alpha = 8)
