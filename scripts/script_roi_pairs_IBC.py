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


# maybe a useless function, run get summary and run_regress individually, more generalizable. keep for now
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
    df1 = ra.get_summary(dataset= dataset, ses_id= ses_id,type=type, add_rest= True,
                            cerebellum_roi=f'tpl-SUIT/atl-{roi_1}_space-SUIT', cortex_roi= f'tpl-fs32k/{roi_1}.32k')

    # Clean up first dataframe (IBC-speciifc)- make this function general
    df1_noprob = df1[df1['cond_name'] != 'probe']
    df1_condall = df1_noprob.groupby(['sn', 'cond_name']).mean()
    df1_condall = df1_condall.reset_index()

    # Run regression for first dataframe
    summary_1 = ra.run_regress(df=df1_condall, fit_intercept= True)

    # Get second dataframe 
    df2 = ra.get_summary(dataset= dataset, ses_id= ses_id,type=type, add_rest= True,
                            cerebellum_roi=f'tpl-SUIT/atl-{roi_2}_space-SUIT', cortex_roi= f'tpl-fs32k/{roi_2}.32k')
    
    # Clean up second dataframe (IBC-speciifc)- make this function general
    df2_noprob = df2[df2['cond_name'] != 'probe']
    df2_condall = df1_noprob.groupby(['sn', 'cond_name']).mean()
    df2_condall = df2_condall.reset_index()

    # Run regression for second dataframe
    summary_2 = ra.run_regress(df=df2_condall, fit_intercept= True)


    return summary_1, summary_2

    
def sim_summary_pair(slope=1.0, seed_value=None):
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



def make_scatterplots(dataframe, split='cond_num', labels=None,
        colors=None,markers=None, ax= None, title = 'plot'):
    """
    make scatterplot
    Args: 
    dataframe (pd.DataFrame) - 
            entire dataframe with individual subject data and fitted slopes and intercepts
    split (str) - column name indicating the different conditions to be plotted
    labels(dict)    - column name to be used to determine shape of the marker
    label (str)    - column name to be used to determine the label of the data points
    height (int)   - int to determine the height of the plot
    aspect (float) - floating number to determine the aspect ratio of the plot
    """
    # do the scatter plot
    grouped = dataframe.groupby([split])
    agg_kw = {split:'first',
              'X':np.mean,'Y': np.mean,
             'slope':np.mean,
             'intercept':np.mean}
    df = grouped.agg(agg_kw)
    
    df["Y_CI"] = grouped.Y.apply(sps.sem) * 1.96
    df["X_CI"] = grouped.X.apply(sps.sem)*1.96
    df['X_err'] = grouped.res.apply(sps.sem)*1.96

    # add  the appropriate errorbars  
    ax.errorbar(x = df['X'], 
                 y = df['Y'], 
                 yerr = df['X_err'],
                 elinewidth=2, 
                fmt='none', # no marker will be used when plotting the error bars
                color=(0.3,0.3,0.3), 
                ecolor=(0.5,0.5,0.5)
                )

    # Plot average regression line 
    xrange = np.array([df['X'].min(),df['X'].max()])
    ypred = xrange*df.slope.mean()+df.intercept.mean()
    ax.plot(xrange,ypred,'k-')

    # Make scatterplot, determining the markers and colors from the dictionary 
    ax = sns.scatterplot(data=df, x='X', y='Y', style = split, hue = split, s = 100,legend= True,markers=markers,palette=colors, ax=ax)

    # set labels
    ax.set_xlabel('Cortical Activation (a.u.)')
    ax.set_ylabel('Cerebellar Activation (a.u.)')
    ax.set_title(title)
    
    return

def make_scatterplots_shift(dataframe_1,dataframe_2, type = 'cerebellum', split='cond_num', labels=None,
        colors=None,markers=None, ax= None):
        grouped_1 = dataframe_1.groupby([split])
        grouped_2 = dataframe_2.groupby([split])
        agg_kw = {split:'first',
              'X':np.mean,'Y': np.mean,
             'slope':np.mean,
             'intercept':np.mean}
        df_1 = grouped_1.agg(agg_kw)
        df_2 = grouped_2.agg(agg_kw)

        df_1["Y_CI"] = grouped_1.Y.apply(sps.sem) * 1.96
        df_1["X_CI"] = grouped_1.X.apply(sps.sem)*1.96

        df_2["Y_CI"] = grouped_2.Y.apply(sps.sem) * 1.96
        df_2["X_CI"] = grouped_2.X.apply(sps.sem)*1.96


        if type == 'cortex':

            # add  the appropriate errorbars  
            ax.errorbar(x = df_1['X'], 
                        y = df_2['X'], 
                        yerr = df_2['X_CI'],
                        xerr = df_1['X_CI'],
                        elinewidth=2, 
                        fmt='none', # no marker will be used when plotting the error bars
                        color=(0.3,0.3,0.3), 
                        ecolor=(0.5,0.5,0.5),
                        alpha = 0.5
                        )
        
            # Make scatterplot, determining the markers and colors from the dictionary 
            ax = sns.scatterplot (x=df_1['X'], y=df_2['X'], style = df_1[split], hue = df_1[split], s = 100,legend= True,markers=markers,palette=colors, ax=ax)
            
            # set labels
            ax.set_xlabel('Language ROI activation (a.u.)')
            ax.set_ylabel('Multi-Demand ROI activation (a.u.)')
            ax.set_title ('Cortex')


        else:
            ax.errorbar(x = df_1['Y'], 
                        y = df_2['Y'], 
                        yerr = df_2['Y_CI'],
                        xerr = df_1['Y_CI'],
                        elinewidth=2, 
                        fmt='none', # no marker will be used when plotting the error bars
                        color=(0.3,0.3,0.3), 
                        ecolor=(0.5,0.5,0.5),
                        alpha = 0.5
                        )
            # Make scatterplot, determining the markers and colors from the dictionary 
            ax = sns.scatterplot (x=df_1['Y'], y=df_2['Y'], style = df_1[split], hue = df_1[split], s = 100,legend= True,markers=markers,palette=colors, ax=ax)

            # set labels
            ax.set_xlabel('Language ROI activation (a.u.)')
            ax.set_ylabel('Multi-Demand ROI activation (a.u.)')
            ax.set_title ('Cerebellum')



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
    make_scatterplots(dataframe_1, ax = ax, split='cond_name', markers= markers, labels=labels, colors=colors, title='Language ROI')
    ax = axes[1,0]
    make_scatterplots(dataframe_2, ax = ax, split='cond_name', markers= markers, labels=labels, colors=colors, title='Multi-Demand ROI')
    ax = axes[0,1]
    make_scatterplots_shift(dataframe_1=dataframe_1,dataframe_2=dataframe_2,
                                    markers= markers, labels=labels, colors=colors, split='cond_name', type='cerebellum', ax=ax)
    ax = axes[1,1]
    make_scatterplots_shift(dataframe_1=dataframe_1,dataframe_2=dataframe_2,
                                    markers= markers, labels=labels, colors=colors, split='cond_name', type='cortex', ax=ax)
    return


