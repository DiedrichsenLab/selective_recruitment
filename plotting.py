import os
import numpy as np
import seaborn as sns # for plots
import nibabel as nb
import nitools as nt
import selective_recruitment.globals as gl
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pathlib import Path
from SUITPy import flatmap
import matplotlib.pyplot as plt
from scipy import stats as sps # to calcualte confidence intervals, etc
from adjustText import adjust_text # to adjust the text labels in the plots (pip install adjustText)

from statsmodels.stats.anova import AnovaRM # perform F test

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
def annotate(dataframe, labels = 'cond_num', text_size = 'small', text_weight = 'regular'):
    """
    annotate data points in the scatterplot
    Args:
    dataframe (pd.DataFrame)
    labels (str,series) - column of the dataframe that is to be used as label
        or dict that maps the index of the dataframe to a label
    text_size (str) 
    text_weight (str)
    labels (str) 
    """
    texts = []
    if labels is str:
        labels = dataframe[labels]
    for i,d in dataframe.iterrows():   
        text = plt.text(
                        d.X+0.001, 
                        d.Y, 
                        s = labels.loc[i],
                        horizontalalignment='left', 
                        size=text_size, 
                        weight=text_weight
                        )
        texts.append(text)

    adjust_text(texts) # make sure you have installed adjust_text

def plot_parcellation(parcellation, roi_name):
    """
    plot the selected region from parcellation on flatmap
    Args:
        parcellation (str) - name of the parcellation
        roi_name (str) - name of the roi as stored in the lookup table
    Return:
        ax (axes object)
        roi_num (int) - number corresponding to the region
    """
    fname = gl.atlas_dir + f'/tpl-SUIT/atl-{parcellation}_space-SUIT_dseg.nii'
    img = nb.load(fname)
    # map it from volume to surface
    img_flat = flatmap.vol_to_surf([img], stats='mode', space = 'SUIT')

    # get the lookuptable for the parcellation
    lookuptable = nt.read_lut(gl.atlas_dir + f'/tpl-SUIT/atl-{parcellation}.lut')

    # get the label info
    label_info = lookuptable[2]
    if '0' not in label_info:
        # append a 0 to it
        label_info.insert(0, '0')
    cmap = LinearSegmentedColormap.from_list("color_list", lookuptable[1])

    # get the index for the region
    roi_num = label_info.index(roi_name)
    roi_flat = img_flat.copy()
    # convert non-selected labels to nan
    roi_flat[roi_flat != float(roi_num)] = np.nan
    # plot the roi
    ax = flatmap.plot(roi_flat, render="plotly", 
                      hover='auto', colorbar = False, 
                      bordersize = 1.5, overlay_type='label', 
                      label_names=label_info, cmap = cmap)

    return ax, roi_num

def make_scatterplot_depricated(dataframe, hue = "phase", style = "recall", label = "load", height = 4, aspect = 1):
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

def make_scatterplot(dataframe, split='cond_num', fit_line = True, labels=None,
        colors=None,markers=None):
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
    plt.errorbar(x = df['X'], 
                 y = df['Y'], 
                 yerr = df['X_err'],
                 elinewidth=2, 
                fmt='none', # no marker will be used when plotting the error bars
                color=(0.3,0.3,0.3), 
                ecolor=(0.5,0.5,0.5)
                )

    # Plot average regression line 
    if fit_line:
        xrange = np.array([df['X'].min(),df['X'].max()])
        ypred = xrange*df.slope.mean()+df.intercept.mean()
        plt.plot(xrange,ypred,'k-')

    # Make scatterplot, determining the markers and colors from the dictionary 
    ax = sns.scatterplot(data=df, x='X', y='Y', style = split, hue = split, s = 100,legend=None,markers=markers,palette=colors)

    # set labels
    ax.set_xlabel('Cortical Activation (a.u.)')
    ax.set_ylabel('Cerebellar Activation (a.u.)')

    # get labels for each data point
    annotate(df, 
            text_size = 'small', 
            text_weight = 'regular', 
            labels = df[split].map(labels))
    return




def annotate2(dataframe, xlabel, ylabel, labels = 'cond_num', text_size = 'small', text_weight = 'regular'):
    """
    annotate data points in the scatterplot
    Args:
    dataframe (pd.DataFrame)
    labels (str,series) - column of the dataframe that is to be used as label
        or dict that maps the index of the dataframe to a label
    text_size (str) 
    text_weight (str)
    labels (str) 
    """
    texts = []
    if labels is str:
        labels = dataframe[labels]
    for i,d in dataframe.iterrows():   
        text = plt.text(
                        d[xlabel]+0.001, 
                        d[ylabel], 
                        s = labels.loc[i],
                        horizontalalignment='left', 
                        size=text_size, 
                        weight=text_weight
                        )
        texts.append(text)

    adjust_text(texts) # make sure you have installed adjust_text
def make_scatterplot2(dataframe, xlabel, ylabel, xerr, yerr,  split='cond_num', labels=None,
        colors=None,markers=None):
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
              xlabel:np.mean,ylabel: np.mean}
    df = grouped.agg(agg_kw)
    
    df[f"{xlabel}_CI"] = grouped[xlabel].apply(sps.sem) * 1.96
    df[f"{ylabel}_CI"] = grouped[ylabel].apply(sps.sem)*1.96
    df['xerr'] = grouped[xerr].apply(sps.sem)*1.96
    df['yerr'] = grouped[yerr].apply(sps.sem)*1.96

    # add  the appropriate errorbars  
    plt.errorbar(x = df[xlabel], 
                 y = df[ylabel], 
                 yerr = df[yerr],
                 xerr = df[xerr],
                 elinewidth=2, 
                 fmt='none', # no marker will be used when plotting the error bars
                 color=(0.3,0.3,0.3), 
                 ecolor=(0.5,0.5,0.5)
                )

    # # Plot average regression line 
    # xrange = np.array([df[xlabel].min(),df['X'].max()])
    # ypred = xrange*df.slope.mean()+df.intercept.mean()
    # plt.plot(xrange,ypred,'k-')

    # Make scatterplot, determining the markers and colors from the dictionary 
    ax = sns.scatterplot(data=df, x=xlabel, y=ylabel, style = split, hue = split, s = 100,legend=None,markers=markers,palette=colors)

    # set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # get labels for each data point
    annotate2(df, xlabel=xlabel, ylabel=ylabel,
            text_size = 'small', 
            text_weight = 'regular', 
            labels = df[split].map(labels))
    return