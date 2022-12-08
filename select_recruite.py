# import packages
import sys
sys.path.append('../Functional_Fusion') 
sys.path.append('../cortico-cereb_connectivity') 

import numpy as np
import pandas as pd
from pathlib import Path

from atlas_map import *
from dataset import *



# WHAT TO DO?
# create an instance of the dataset 

# extract the data within atlas for cerebellum

# extract the data within atlas for cortex

# extract data within selected parcellation for cerebellum

# extract data within selected parcellation for cortex

# regress cerebellar data onto cortical data 
    # get residuals

# get summary (a wrapper)

# set the directory of your dataset here:
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'

data_dir = base_dir + '/WMFS'
atlas_dir = base_dir + '/Atlases'

def extract_suit(dataSet, ses_id, type, atlas):
    # create an instance of the dataset class
    dataset = dataSet(data_dir)

    # extract data for suit atlas
    dataset.extract_all_suit(ses_id,type,atlas)
    pass
def extract_fs32K(dataSet, ses_id, type):
    # create an instance of the dataset class
    dataset = dataSet(data_dir)

    # extract data for suit atlas
    dataset.extract_all_fs32k(ses_id,type)
    pass

def get_summary(data_dir, ses_id = 2, type = 'CondAll'):

    """
    Args:
    dataset (class dataset) -  an instance of Dataset class in functional fusion
    """
    
    # extract cerebellar data
    extract_suit(dataSet, ses_id, type, atlas)

    # extract cortical data
    extract_fs32K(dataSet, ses_id, type)

    return
