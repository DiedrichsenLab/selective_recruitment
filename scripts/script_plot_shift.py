"""
script to prepare dataframe cerebellar shifts
@ Bassel Arafat Feb 17th 2023 11:28
"""
import os
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict

import nibabel as nb
import Functional_Fusion.dataset as fdata 
import Functional_Fusion.atlas_map as am
import cortico_cereb_connectivity.data as cdata 
import selective_recruitment.recruite_ana as ra

# set base directory of the functional fusion 
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
atlas_dir = '/cifs/diedrichsen/data/Cerebellum/Language/atlases'

# run this function to make sure that you have saved data tensor
def prep_tensor(dataset_name = "WMFS", ses_id = 'ses-02'):
   cdata.save_data_tensor(dataset = dataset_name,
                         atlas='SUIT3',
                         ses_id=ses_id,
                         type="CondHalf")

# prepare data for cortex and cerebellum
def get_XY_cereb(label_cereb_x, 
           label_cereb_y,
           file_path = cdata.conn_dir, 
           dataset = "WMFS", 
           ses_id = "ses-02",
           type = "CondHalf", 
           atlas_cereb = "SUIT3",
           unite_struct = False
           ):
    # get dataset class object
    Data = fdata.get_dataset_class(base_dir, dataset=dataset)

    # get info
    info = Data.get_info(ses_id,type)

    # load data tensor for SUIT3 (TODO: add an option to specify atlases other than SUIT3 and fs32k)
    file_suit = file_path + f'/{dataset}/{dataset}_SUIT3_{ses_id}_{type}.npy'
    Ydat = np.load(file_suit)
    
    # create instance of atlase for each cerebellar roi ??
    atlas_cereb_x, ainfo = am.get_atlas('SUIT3',atlas_dir)
    atlas_cereb_y, ainfo = am.get_atlas('SUIT3',atlas_dir)

    
    
    # get parcel for both atlases
    atlas_cereb_x.get_parcel(label_cereb_x)
    atlas_cereb_y.get_parcel(label_cereb_y)


    # aggregate data over parcels
    x_parcel = fdata.agg_parcels(Ydat , atlas_cereb_x.label_vector,fcn=np.nanmean)
    y_parcel = fdata.agg_parcels(Ydat , atlas_cereb_y.label_vector,fcn=np.nanmean)

    return x_parcel, y_parcel, info

    

def get_summary_shift(outpath = cdata.conn_dir, 
                    dataset = "WMFS", 
                    ses_id = 'ses-02', 
                    cerebellum_x = "Verbal2Back", 
                    cerebellum_y = "Verbal2Back", 
                    type = "CondHalf"):

    """
    get summary dataframe for roi shift analysis analysis
    """

    # get label files for the two cerebellar rois 
    # NOTE: To average over cerebellum or cortex, pass on masks as label files
    label_cereb_x = atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum_x}_space-SUIT_dseg.nii'
    label_cereb_y = atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum_y}_space-SUIT_dseg.nii'

     

    x_parcel, y_parcel, info = get_XY_cereb(label_cereb_x, 
                                      label_cereb_y,
                                      file_path = cdata.conn_dir, 
                                      dataset = dataset, 
                                      ses_id = ses_id,
                                      atlas_cereb = "SUIT3",
                                      type = type, 
                                      unite_struct=True
                                     )

    summ_df = ra.run_regress(x_parcel,y_parcel,info,fit_intercept = False)

    # save the dataframe
    filepath = os.path.join(outpath, dataset, f'sc_{dataset}_{ses_id}_{cerebellum_x}-vs-{cerebellum_y}-shift.tsv')
    summ_df.to_csv(filepath, index = False, sep='\t')
    return


if __name__ == "__main__":
    # prep_tensor(dataset_name = "WMFS", ses_id = "ses-01")
    
    get_summary_shift(dataset= 'IBC',ses_id='ses-rsvplanguage', cerebellum_x='ff_languageLR',cerebellum_y='ff_wmLR', type='condhalf')