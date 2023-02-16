import numpy as np
import nibabel as nb
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from pathlib import Path


base_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/Language'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/Cerebellum/Language'
if not Path(base_dir).exists():
    base_dir = '//bmisrv.robarts.ca/Diedrichsen_data$/data/Cerebellum/Language'
if not Path(base_dir).exists():
    base_dir = '/cifs/diedrichsen/data/Cerebellum/Language'


def locdata_to_csv(sub_list,conditions,loc,dest_name, threshold=0.1, percentile=False):
    mask = nb.load(
        f'{base_dir}/cortical_masks/cortical_mask_gm_binary_MNI_2mm.nii')
    cortical_mask = mask.get_fdata()
    mask = nb.load(
        f'{base_dir}/cerebellar_masks/cerebellar_mask_gm_binary_MNI_2mm.nii')
    cerebellar_mask = mask.get_fdata()

    df={'sub_id':[],'sn':[],'CN':[],'CN_#':[],'cortex':[],'cerebellum':[]}

    thresh = threshold
    for i, sub in enumerate(sub_list) :

        # load localizer image
        loc_mask = nb.load(
            f'{base_dir}/derivatives/raw/sub-{sub}/sub-{sub}_{loc}.nii')
        loc_data = loc_mask.get_fdata()

        # calculate threshold value for percentile option (only the first time you loop through inner loop)
        if percentile:
            thresh = np.percentile(loc_data, 100-threshold)

        # sanity check: print percentage of voxels included
        percentage = len(
            loc_data[loc_data > thresh].flatten()) / len(loc_data.flatten())
        print('Included  {:.0%}'.format(percentage))

        # threshold localizer
        loc_data[loc_data < thresh] = 0
        loc_data[loc_data > thresh] = 1

        for x,condition in enumerate(conditions):
            
            # append info
            df['sub_id'].append(sub)
            df['sn'].append(i)
            df['CN'].append(condition) 
            assigned_condition_number=x
            df['CN_#'].append(assigned_condition_number)

            # load contrast image
            con_image = nb.load(
                f'{base_dir}/raw/sub-{sub}/sub-{sub}_{condition}.nii')
            con_data=con_image.get_fdata()            

            # mask contrast by localizer
            LL_con_data= con_data * loc_data
            cortical_map= LL_con_data * cortical_mask
            cerebellar_map= LL_con_data * cerebellar_mask

            # append data
            df['cortex'].append(np.nanmean(cortical_map.flatten()))
            df['cerebellum'].append(np.nanmean(cerebellar_map.flatten()))

    d= pd.DataFrame(df)
    d.to_csv(f'{base_dir}/{dest_name}')



if __name__ == "__main__":

    image_type = 't' # 't' or 'c'on for t-stat image or contrast image (beta values)

    subs = ('261', '742', '746', '747', '748', '749', '750', '751', '752', '753', '754', '755', '756', '757',
            '758', '770', '788', '797', '803', '805', '806', '807', '824', '825', '826', '827', '828', '831')
    
    conditions = ('ProdE1_NProd', 'ProdE1_SComp',
                     'ProdE1_SProd', 'ProdE1_WComp', 'ProdE1_WProd')
    conditions = [f'{condition}_{image_type}' for condition in conditions]

    # Comprehension-based localizer
    locdata_to_csv(sub_list=subs, conditions=conditions, loc=f'langloc_S-N_{image_type}',
                   dest_name=f'language_data_analyses/langloc_comp/comp_{image_type}.csv', threshold=0)

    # Production-based localizer
    locdata_to_csv(sub_list=subs, conditions=conditions, loc=f'prod_langloc_S-N_{image_type}',
                   dest_name=f'language_data_analyses/langloc_prod/prod_{image_type}.csv', threshold=0)

    # reproducing fedorenko localizer approach (comprehension localizer)
    # locdata_to_csv(sub_list=subs, conditions=conditions, loc=f'langloc_S-N_{image_type}',
    #                dest_name=f'language_data_analyses/langloc_comp/comp_perc_{image_type}.csv', threshold=10, percentile=True)

    pass






