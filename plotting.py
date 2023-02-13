import os
import numpy as np
import seaborn as sns # for plots
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats as sps # to calcualte confidence intervals, etc
from adjustText import adjust_text # to adjust the text labels in the plots (pip install adjustText)

from statsmodels.stats.anova import AnovaRM # perform F test

# Functions
# prepare dataframe for plotting
def prep_df(dataframe, agg_kw = {}, error = 'res', groupby = "cond_name"):
    """
    prepare the region dataframe to do the scatter plot
    gets the mean across subjects (data point) and std of residuals
    THIS ONLY WORKS FOR WM dataset
    Args:
        dataframe (pd.DataFrame) - dataframe with residuals info 
        agg_kw (dict) - dictionary determining info for dataframe aggregation
        Example agg_kw: {'load': 'first',
                         'phase': 'first',
                         'recall': 'first',
                         'X': np.mean,
                         'Y': np.mean}
    Returns:
    g_df (pd.DataFrame) - dataframe ready for putting into the scatterplot function
    """
    # group by condition
    grouped = dataframe.groupby([groupby])
    g_df = grouped.agg(agg_kw)
    
    g_std = grouped.std(numeric_only=True)
    g_df["Y_CI"] = grouped.Y.apply(sps.sem) * 1.96
    g_df["X_CI"] = grouped.X.apply(sps.sem)*1.96
    g_df['err'] = g_std[error]
    
    
    return g_df

# add text labels to points
def annotate(dataframe, text_size = 'small', text_weight = 'regular', labels = "load"):
    """
    annotate data points in the scatterplot
    Args:
    dataframe (pd.DataFrame)
    text_size (str) 
    text_weight (str)
    labels (str) - column of the dataframe that is to be used as label
    """
    texts = []
    for i in range(dataframe.shape[0]):   
        text = plt.text(
                        dataframe.X[i]+0.001, 
                        dataframe.Y[i], 
                        s = dataframe[labels][i],
                        horizontalalignment='left', 
                        size=text_size, 
                        weight=text_weight
                        )
        texts.append(text)

    adjust_text(texts) # make sure you have installed adjust_text

# plotting:
def make_scatterplot(dataframe, hue = "phase", style = "recall", label = "load", height = 4, aspect = 1):
    """
    make scatterplot
    uses FacetGrid 
    Args: 
    dataframe (pd.DataFrame) - output from prep_df
    hue (str)      - column name to be used to determine color
    style (str)    - column name to be used to determine shape of the marker
    label (str)    - column name to be used to determine the label of the data points
    height (int)   - int to determine the height of the plot
    aspect (float) - floating number to determine the aspect ratio of the plot
    """
    g = sns.FacetGrid(dataframe,  height=height, aspect=aspect)
    # do the scatter plot
    g.map_dataframe(sns.scatterplot, x="X", y="Y", 
                                    style = style, hue = hue, s = 100)
    g.add_legend()

    # fit the regression on top of the scatterplot
    g.map_dataframe(sns.regplot, x="X", y="Y", 
                        fit_reg=True, 
                        scatter_kws={"s": 0}, # size is set to 0 so that it doesn't cover the markers created in the scatterplot step 
                        line_kws={'label':"Linear Reg", "color": 'grey'})

    # put the errorbars in 
    g.map(plt.errorbar, x = dataframe['X'], 
                        y = dataframe['Y'], 
                        yerr = dataframe['err'],
                        elinewidth=1, 
                        fmt='none', # no marker will be used when plotting the error bars
                        color='grey', 
                        ecolor='0.9'
                )
    # set labels
    g.set_xlabels('Cortical Activation (a.u.)')
    g.set_ylabels('Cerebellar Activation (a.u.)')

    # get labels for each data point
    annotate(dataframe, text_size = 'small', text_weight = 'regular', labels = label)
    return