# script to do scatter plot of whole cortex vs whole cerebellum
# Ladan shahshahani
# BIDS format/directory structure

# base_dir = <project directory>

# packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nb
import nitools as niu
import os

base_dir = os.path.join('C:\\Users\\lshah\\OneDrive\\Documents\\Data\\FunctionalFusion\\WMFS')
deriv_dir = os.path.join(base_dir, 'derivatives')




class Data():
    
    def __init__(self, subj_id, ses, atlas, integ_type):
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
    def get_data(self):
        """
        extract data for the region of interest
        """                                   
        info_file = os.path.join(deriv_dir, self.subj_id, 'data', f'{self.subj_id}_ses-{self.ses:02d}_info-{self.type}.tsv')
        self.info = pd.read_csv(info_file, sep='\t')
        data_file = os.path.join(deriv_dir, self.subj_id, 'data', f'{self.subj_id}_space-{self.atlas}_ses-{self.ses:02d}_{self.type}.dscalar.nii')
        data_cifti = nb.load(data_file)
        self.data = data_cifti.get_fdata()
        print(self.data.shape)
        bmf = data_cifti.header.get_axis(1)
        data_list = []
        for idx, (nam,slc,bm) in enumerate(bmf.iter_structures()):
            print(idx,str(nam),slc)
    def get_average():
        """
        average data across the region of interest
        uses self.info
        """

        pass
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

def data_df(xdata, ydata, info):
    """
    prepare data for scatter plot.
    Place average across one roi on x axis and one 
    on the y axis.
    xdata (Data class)
    ydata (Data class)
    info (info dataframe)
    """
    # get the average across ROI
    x = xdata.get_average()
    y = ydata.get_average()

    # create a dataframe for 

    return


if __name__ == "__main__":
    D_suit3 = Data(subj_id='sub-01', ses = 2, atlas = 'SUIT3', integ_type='CondHalf')
    D_suit3.get_data()
    
    D_fs32k = Data(subj_id='sub-01', ses = 2, atlas = 'fs32k', integ_type='CondHalf')
    D_fs32k.get_data()


    print('hello')
