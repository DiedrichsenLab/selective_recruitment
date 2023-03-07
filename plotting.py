import os
import numpy as np
import seaborn as sns # for plots
import pandas as pd
from pathlib import Path
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

def make_scatterplot(dataframe, split='cond_num', labels=None,
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

    # # get labels for each data point
    # annotate(df, 
    #         text_size = 'small', 
    #         text_weight = 'regular', 
    #         labels = df[split].map(labels))
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

        if type == 'cortex':
            # ax.errorbar(x = df_1['X'], 
            #      y = df_2['X'],
            #      xerr = df_1['1_X_CI'], 
            #      yerr = df_2['2_X_CI'],
            #      elinewidth=2, 
            #     fmt='none', # no marker will be used when plotting the error bars
            #     color=(0.3,0.3,0.3), 
            #     ecolor=(0.5,0.5,0.5)
                # )
        
            # Make scatterplot, determining the markers and colors from the dictionary 
            ax = sns.scatterplot (x=df_1['X'], y=df_2['X'], style = df_1[split], hue = df_1[split], s = 100,legend= True,markers=markers,palette=colors, ax=ax)
            
            # set labels
            ax.set_xlabel('Language ROI activation (a.u.)')
            ax.set_ylabel('Multi-Demand ROI activation (a.u.)')
            ax.set_title ('Cortex')

        else:
            # ax.errorbar(x = df_1['Y'], 
            #      y = df_2['Y'],
            #      xerr = df_1['1_Y_CI'], 
            #      yerr = df_2['2_Y_CI'],
            #      elinewidth=2, 
            #     fmt='none', # no marker will be used when plotting the error bars
            #     color=(0.3,0.3,0.3), 
            #     ecolor=(0.5,0.5,0.5)
                # )
        
            # Make scatterplot, determining the markers and colors from the dictionary 
            ax = sns.scatterplot (x=df_1['Y'], y=df_2['Y'], style = df_1[split], hue = df_1[split], s = 100,legend= True,markers=markers,palette=colors, ax=ax)

            # set labels
            ax.set_xlabel('Language ROI activation (a.u.)')
            ax.set_ylabel('Multi-Demand ROI activation (a.u.)')
            ax.set_title ('Cerebellum')
            





