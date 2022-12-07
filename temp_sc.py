# script to do scatter plot of whole cortex vs whole cerebellum
# Ladan shahshahani
# BIDS format/directory structure

# base_dir = <project directory>

# adding functional fusion package
import sys
sys.path.append('C:\\Users\\lshah\\OneDrive\\Documents\\Projects\\Functional_Fusion') 

# packages
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nb
# import nitools as niu

from atlas_map import *

base_dir = os.path.join('C:\\Users\\lshah\\OneDrive\\Documents\\Data\\FunctionalFusion\\WMFS')
deriv_dir = os.path.join(base_dir, 'derivatives')




class Data():
    
    def __init__(self, subj_id, ses, atlas, integ_type = 'CondHalf'):
        """
        Data class to be used when infering selective recruitment
        Args:
        atlas (str) - name of the region of interest
        info (pd.DataFrame) - dataframe containing information about task/conditions
        integ_type (str) - how to integrate data across runs/sessions
        """
        self.subj_id = subj_id
        self.ses = ses
        self.atlas = atlas
        self.type = integ_type
        
    def load_data(self):
        """
        extract data for the region of interest
        """                                   
        info_file = os.path.join(deriv_dir, self.subj_id, 'data', f'{self.subj_id}_ses-{self.ses:02d}_info-{self.type}.tsv')
        self.info = pd.read_csv(info_file, sep='\t')
        data_file = os.path.join(deriv_dir, self.subj_id, 'data', f'{self.subj_id}_space-{self.atlas}_ses-{self.ses:02d}_{self.type}.dscalar.nii')
        data_cifti = nb.load(data_file)
        self.data = data_cifti.get_fdata()
    
    def get_parcel_vol_data(self, label_img, mask_img = None, name = 'cerebellum'):
        """
        get data within parcels defined in label image
        constraining it to the mask image
        Args:
        label_img (str)
        mask_img(str)
        """
        parcel_atlas = AtlasVolumeParcel(name,label_img,mask_img)
        self.get_parcel_data= parcel_atlas.agg_data(self.data,func=np.nanmean)

    def get_parcel_surf_data(self, label_img, mask_img = None, name = 'cortex_left'):
        """
        get data within parcels defined in label image
        constraining it to the mask image
        Args:
        label_img (str)
        mask_img(str)
        """
        parcel_atlas = AtlasSurfaceParcel(name,label_img,mask_img)
        self.get_parcel_data= parcel_atlas.agg_data(self.data,func=np.nanmean)
        
    def get_data(self):
        """
        method to extract data using functional fusion 
        package if it hasn't been done

        UNDER CONSTRUCTION
        """
    
    def get_average(self):
        """
        average data across the region of interest
        uses self.info
        """
        self.atlas_data = np.nanmean(self.data, axis = 1)
    
    def get_group_average(self, sn):
        """
        calculates group average data 
        Args:
        sn (list) - list of subject ids in the group
        Returns:
        group average dataset
        """
        pass

def get_summary(subj_list = [], atlas_list = [], parcels = {'cerebellum': 'mdtb10'}):
    """
    prepare data for scatter plot.
    Place average across one roi on x axis and one 
    on the y axis.
    Args:
    subj_list (list) - list of subjects. Example: subj_list = ['sub-01', 'sub-02']
    atlas_list (list) - list of atlases you want to get the data for. Example: atlas_list = ['SUIT3', 'fs32k']
    Returns:
    df (pd.DataFrame) - dataframe containing the summary of data to do the scatterplot
    """
    df = pd.DataFrame()
    for atlas in atlas_list:
        for s in subj_list:
            print(f"- Doing {s} in {atlas}")
            # create instances of the data class
            D = Data(subj_id=s, ses = 2, atlas = atlas, integ_type='CondHalf')
            # get data
            D.load_data()
            # get the average over the roi
            D.get_average()
            dd = D.info.copy()
            dd['value'] = D.atlas_data
            df = df.append(dd, ignore_index = True)

    return df


if __name__ == "__main__":
    df = get_summary(['sub-01'], atlas_list = ['SUIT3', 'fs32k'])

"""
Example usages:
# creating a Data class for SUIT 3mm resolution for sub-01
D_suit3 = Data(subj_id='sub-01', ses = 2, atlas = 'SUIT3', integ_type='CondHalf')

# reading the data extracted and saved as cifti 
Dsuit3.load_data()

# calculating average over the atlas
Dsuit3.get_average()

# inspecting the Data class created for suit 3 mm resolution
print(Dsuit3.atlas_data.shape) # averaged data over the region (atlas?)
print(Dsuit3.data.shape) # data loaded in
print(Dsuit3.info) # information for the tasks/condition
"""
