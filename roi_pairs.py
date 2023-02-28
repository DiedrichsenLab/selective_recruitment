"""
The functions used to do the selective recruitment - shifting analysis

Created on 02/27/2023 at 10:35 am
Author: Bassel Arafat
"""

from cProfile import label
import os
from turtle import color
from matplotlib import markers
import numpy as np
import seaborn as sns # for plots
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats as sps # to calcualte confidence intervals, etc
from adjustText import adjust_text
from plotting import make_scatterplot # to adjust the text labels in the plots (pip install adjustText)
import selective_recruitment.globals as gl
import recruite_ana as ra
import selective_recruitment.plotting as plotting

def get_summary_pair(dataset = 'IBC',
                    ses_id = 'ses-rsvplanguage',
                    type = 'CondAll',
                    add_rest = False, 
                    roi_1 ='ff_lang', 
                    roi_2 = 'ff_wm'):
    """
    Get dataframe pair for two rois. Regression is done and included in the dataframe

    Returns:
    Summary_1: dataframe 1
    Summary_2: dataframe 2

    """
    # Get first dataframe 
    df1 = ra.get_summary(dataset= dataset, ses_id= ses_id,type=type, add_rest= False,
                            cerebellum_roi=f'tpl-SUIT/atl-{roi_1}_space-SUIT', cortex_roi= f'tpl-fs32k/{roi_1}.32k')

    # Clean up first dataframe (IBC-speciifc)- make this function general
    df1_noprob = df1[df1['cond_name'] != 'probe']
    df1_condall = df1_noprob.groupby(['sn', 'cond_name']).mean()
    df1_condall = df1_condall.reset_index()

    # Run regression for first dataframe
    summary_1 = ra.run_regress(df=df1_condall, fit_intercept= True)

    # Get second dataframe 
    df2 = ra.get_summary(dataset= dataset, ses_id= ses_id,type=type, add_rest= False,
                            cerebellum_roi=f'tpl-SUIT/atl-{roi_2}_space-SUIT', cortex_roi= f'tpl-fs32k/{roi_2}.32k')
    
    # Clean up second dataframe (IBC-speciifc)- make this function general
    df2_noprob = df2[df2['cond_name'] != 'probe']
    df2_condall = df2_noprob.groupby(['sn', 'cond_name']).mean()
    df2_condall = df2_condall.reset_index()

    # Run regression for second dataframe
    summary_2 = ra.run_regress(df=df2_condall, fit_intercept= True)


    return summary_1, summary_2

    



def sim_summary_pair(x_min=-0.11380856779352279, x_max=0.04603078849976064,
                     y_min=-0.10749953012802338, y_max=0.013798874028263913,
                     scale_x1=1, scale_x2=1, scale_y1=1, scale_y2=1, seed_value=None):
    """
    Simulate two dataframes with random data

    Args:
    x_min: min x value
    x_max: max x value
    y_min: min y value
    y_max: max y value
    scale_x1: scaling factor for df1 x
    scale_x2: scaling factor for df2 x
    scale_y1: scaling factor for df1 y
    scale_y2: scaling factor for df2 y
    seed_value: seed for the random generator (default: None)

    Returns:
    df1: DataFrame containing simulated data for dataset 1
    df2: DataFrame containing simulated data for dataset 2
    """
    # Set seed of randomness
    np.random.seed(seed_value)

    # Define subjects and conditions
    subjects = np.arange(13)
    conditions = ['word_list', 'psuedoword_list', 'simple_sentence', 'complex_sentence',
                  'consonant_string', 'jabberwocky']

    # Create empty dataframes
    df1 = pd.DataFrame(columns=['sn', 'cond_name', 'X', 'Y', 'roi'])
    df2 = pd.DataFrame(columns=['sn', 'cond_name', 'X', 'Y', 'roi'])

    # Fill dataframes with simulated data
    for sn in subjects:
        for condition in conditions:
            x1 = np.random.uniform(x_min, x_max)
            x2 = np.random.uniform(x_min, x_max)
            y1 = np.random.uniform(y_min, y_max)
            y2 = np.random.uniform(y_min, y_max)

            # Scale the X and Y values based on input scaling factors
            x_scaled_1 = x1 * scale_x1
            x_scaled_2 = x2 * scale_x2
            y_scaled_1 = y1 * scale_y1
            y_scaled_2 = y2 * scale_y2

            df1 = df1.append({'sn': sn, 'cond_name': condition, 'X': x_scaled_1, 'Y': y_scaled_1, 'roi': 0.0},
                             ignore_index=True)
            df2 = df2.append({'sn': sn, 'cond_name': condition, 'X': x_scaled_2, 'Y': y_scaled_2, 'roi': 0.0},
                             ignore_index=True)

    return df1, df2




def plot_pair(dataframe_1,dataframe_2 , markers= None, labels = None, colors= None):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize= (15,10))
    """
    Plot the 4 plots for the shifting analysis. Outputs a figure with 4 plots.

    Plot 1 [0,0]: Cerebellum vs Cortex for region 1
    Plot 2 [1,0]: Cerebellum vs Cortex for region 2
    Plot 1 [0,1]: Cereballar activations for region 1 vs region 2
    Plot 1 [1,1]: Cortical activations for region 1 vs region 2

    """

    ax = axes[0,0]
    plotting.make_scatterplots(dataframe_1, ax = ax, split='cond_name', markers= markers, labels=labels, colors=colors, title='Language ROI')
    ax = axes[1,0]
    plotting.make_scatterplots(dataframe_2, ax = ax, split='cond_name', markers= markers, labels=labels, colors=colors, title='Multi-Demand ROI')
    ax = axes[0,1]
    plotting.make_scatterplots_shift(dataframe_1=dataframe_1,dataframe_2=dataframe_2,
                                    markers= markers, labels=labels, colors=colors, split='cond_name', type='cerebellum', ax=ax)
    ax = axes[1,1]
    plotting.make_scatterplots_shift(dataframe_1=dataframe_1,dataframe_2=dataframe_2,
                                    markers= markers, labels=labels, colors=colors, split='cond_name', type='cortex', ax=ax)
    return


