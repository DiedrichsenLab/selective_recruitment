"""
The functions used to do the selective recruitment - shifting analysis

Created on 02/27/2023 at 10:35 am
Author: Bassel Arafat
"""

from cProfile import label
from mmap import MAP_ANONYMOUS
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
                            cerebellum_roi=f'tpl-SUIT/atl-{roi_1}_space-SUIT', cortex_roi= f'tpl-fs32k/{roi_1}.32k', add_rest = True)

    # Clean up first dataframe (IBC-speciifc)- make this function general
    df1_noprob = df1[df1['cond_name'] != 'probe']
    df1_condall = df1_noprob.groupby(['sn', 'cond_name']).mean()
    df1_condall = df1_condall.reset_index()

    # Run regression for first dataframe
    summary_1 = ra.run_regress(df=df1_condall, fit_intercept= True)

    # Get second dataframe 
    df2 = ra.get_summary(dataset= dataset, ses_id= ses_id,type=type, add_rest= False,
                            cerebellum_roi=f'tpl-SUIT/atl-{roi_2}_space-SUIT', cortex_roi= f'tpl-fs32k/{roi_2}.32k', add_rest = True)
    
    # Clean up second dataframe (IBC-speciifc)- make this function general
    df2_noprob = df2[df2['cond_name'] != 'probe']
    df2_condall = df2_noprob.groupby(['sn', 'cond_name']).mean()
    df2_condall = df2_condall.reset_index()

    # Run regression for second dataframe
    summary_2 = ra.run_regress(df=df2_condall, fit_intercept= True)


    return summary_1, summary_2

    
def sim_summary_pair_n(slope=1.0, seed_value=None):
    """
    Simulate a DataFrame with random data and a linear relationship between X and Y controlled by the slope

    Args:
    slope: slope of the linear relationship between X and Y
    seed_value: seed for the random generator (default: None)

    Returns:
    df: DataFrame containing simulated data
    """
    #  seed of randomness
    np.random.seed(seed_value)

    # define subjects and conditions
    subjects = np.arange(13)
    conditions = ['word_list', 'psuedoword_list', 'simple_sentence', 'complex_sentence',
                  'consonant_string', 'jabberwocky']

    #define mean and variance for X
    mx = np.mean(np.random.uniform(size=1000))
    vx = np.var(np.random.uniform(size=1000))

    # define mean and variance for Y
    my = slope * mx
    vy = np.var(np.random.uniform(size=1000))

    # create empty dataframe
    df = pd.DataFrame(columns=['sn', 'cond_name', 'X', 'Y', 'roi'])

    # fill dataframe with simulated data
    for subject in subjects:
        for condition in conditions:
            x = mx + np.random.normal(scale=np.sqrt(vx))
            y = my + np.random.normal(scale=np.sqrt(vy))
            df = df.append({'sn': subject, 'cond_name': condition, 'X': x, 'Y': y ,'roi':0.0}, ignore_index=True)

    return df



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


