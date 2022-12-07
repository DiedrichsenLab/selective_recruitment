# script to do scatter plot of whole cortex vs whole cerebellum
# Ladan shahshahani
# BIDS format/directory structure

# base_dir = <project directory>

# adding functional fusion package
import sys
sys.path.append('../Functional_Fusion') 

# packages
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nb
# import nitools as niu

from atlas_map import *

# base_dir = os.path.join('C:\\Users\\lshah\\OneDrive\\Documents\\Data\\FunctionalFusion\\WMFS')
base_dir = '/Volumes/Diedrichsen_data$/data/FunctionalFusion/WMFS'
deriv_dir = os.path.join(base_dir, 'derivatives')
atlas_fs32k = os.path.join('/Volumes/Diedrichsen_data$/data/FunctionalFusion/Atlases', 'tpl-fs32k')
atlas_suit  = os.path.join('/Volumes/Diedrichsen_data$/data/FunctionalFusion/Atlases', 'tpl-SUIT')



class Cerebellum():
    
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
    
    def agg_data(self, label_img = None, mask_img = None, name = 'cerebellum'):
        """
        get data within parcels defined in label image
        constraining it to the mask image
        Args:
        label_img (str)
        mask_img(str)
        """
        if label_img != None:
            parcel_atlas = AtlasVolumeParcel(name,label_img,mask_img)
            self.parcel_data= parcel_atlas.agg_data(self.data,func=np.nanmean)
        else:
            self.parcel_data = np.nanmean(self.data, axis = 1)

        
    def get_data(self):
        """
        method to extract data using functional fusion 
        package if it hasn't been done

        UNDER CONSTRUCTION
        """
    
    def get_group_average(self, sn):
        """
        calculates group average data 
        Args:
        sn (list) - list of subject ids in the group
        Returns:
        group average dataset
        UNDER CONSTRUCTION
        """
        pass

class Cortex():
    def __init__(self, subj_id, ses, atlas, hemi = 1, integ_type = 'CondHalf'):
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
        self.hemi = hemi
        
    def load_data(self):
        """
        extract data for the region of interest
        """                                   
        info_file = os.path.join(deriv_dir, self.subj_id, 'data', f'{self.subj_id}_ses-{self.ses:02d}_info-{self.type}.tsv')
        self.info = pd.read_csv(info_file, sep='\t')
        data_file = os.path.join(deriv_dir, self.subj_id, 'data', f'{self.subj_id}_space-{self.atlas}_ses-{self.ses:02d}_{self.type}.dscalar.nii')
        data_cifti = nb.load(data_file)
        self.data = data_cifti.get_fdata()
        self.data = self.data[:, (self.hemi-1)*29759: self.hemi*29759 + 29759]

    def agg_data(self, label_img = None, mask_img = None, name = 'cortex_left'):
        """
        get data within parcels defined in label image
        constraining it to the mask image
        Args:
        label_img (str)
        mask_img(str)
        """
        if label_img != None:
            parcel_atlas = AtlasSurfaceParcel(name,label_img,mask_img)
            self.parcel_data= parcel_atlas.agg_data(self.data,func=np.nanmean)
        else:
            self.parcel_data = np.nanmean(self.data, axis = 1)

        
    def get_data(self):
        """
        method to extract data using functional fusion 
        package if it hasn't been done

        UNDER CONSTRUCTION
        """
    
    def get_group_average(self, sn):
        """
        calculates group average data 
        Args:
        sn (list) - list of subject ids in the group
        Returns:
        group average dataset
        """
        pass
    pass

def get_summary_cereb(subj_list = [], parcels = {'SUIT3':[None, 'MDTB10', 'Buckner7', 'Buckner17']}):
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
    for atlas in parcels.keys():

        for parcel in parcels[atlas]:

            for s in subj_list:
                print(f"- Doing {s} in {atlas} {parcel}")
                # create instances of the data class
                D = Cerebellum(subj_id=s, ses = 2, atlas = atlas, integ_type='CondHalf')
                # get data
                D.load_data()
                # get the average over eaxh parcel
                if parcel != None:
                    mask_img = os.path.join(atlas_suit, 'tpl-SUIT_res-3_gmcmask.nii')
                    label_img = os.path.join(atlas_suit, f'atl-{parcel}_space-SUIT_dseg.nii')
                    D.agg_data(label_img=label_img, mask_img=mask_img, name = 'Cerebellum')

                    for region in range(D.parcel_data.shape[1]):
                        dd = D.info.copy()
                        
                        dd['value'] = D.parcel_data[:, region]
                        dd['atlas'] = atlas
                        dd['region'] = region
                        dd['parcellation'] = parcel
                        df = df.append(dd, ignore_index = True)
                else:
                    D.agg_data(label_img=None, mask_img=None, name = 'Cerebellum')
                    dd = D.info.copy()
                        
                    dd['value'] = D.parcel_data
                    dd['atlas'] = atlas
                    dd['region'] = -1
                    dd['parcellation'] = parcel
                    df = df.append(dd, ignore_index = True)
    return df

def get_summary_cortex(subj_list = [], parcels = {'fs32k':[None, 'ROI']}):
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
    names = ['cortex_left', 'cortex_right']
    # hemis = ['L', 'R']
    for atlas in parcels.keys():

        for parcel in parcels[atlas]:

            for s in subj_list:
                print(f"- Doing {s} in {atlas} {parcel}")
                # create instances of the data class
                D = Cortex(subj_id=s, ses = 2, atlas = atlas, integ_type='CondHalf')
                # get data
                D.load_data()

                for h, hemi in enumerate(['L', 'R']):
                # get the average over eaxh parcel
                    if parcel != None:
                        mask_img = os.path.join(atlas_fs32k, f'tpl-fs32k_hemi-{hemi}_mask.label.gii')
                        label_img = os.path.join(atlas_fs32k, f'{parcel}.{hemi}.label.gii')
                        D.agg_data(label_img=label_img, mask_img=mask_img, name = names[h])

                        for region in range(D.parcel_data.shape[1]):
                            dd = D.info.copy()
                            
                            dd['value'] = D.parcel_data[:, region]
                            dd['atlas'] = atlas
                            dd['region'] = region
                            dd['parcellation'] = parcel

                            df = df.append(dd, ignore_index = True)
                    else:
                        mask_img = os.path.join(atlas_fs32k, f'tpl-fs32k_hemi-{hemi}_mask.label.gii')
                        D.agg_data(label_img=mask_img, mask_img=mask_img, name = names[h])
                        dd = D.info.copy()
                            
                        dd['value'] = D.parcel_data
                        dd['atlas'] = atlas
                        dd['region'] = -1
                        dd['parcellation'] = parcel
                        df = df.append(dd, ignore_index = True)


if __name__ == "__main__":
    subj_list = []
    for s in range(1, 17):
        subj_list.append(f'sub-{s:02d}')
    df_cereb = get_summary_cereb(subj_list=subj_list)
    df_cortex = get_summary_cortex(subj_list=subj_list)

    df_final = pd.concat([df_cereb, df_cortex], axis = 0)

    # save the dataframe
    file_name = os.path.join(base_dir, 'summary_sc.csv')
    df_final.to_csv(file_name)

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
