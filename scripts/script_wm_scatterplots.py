import os
import numpy as np
import seaborn as sns # for plots
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats as sps # to calcualte confidence intervals, etc
from adjustText import adjust_text # to adjust the text labels in the plots (pip install adjustText)

import selective_recruitment.plotting as plotting
import selective_recruitment.recruite_ana as ra
import selective_recruitment.globals as gl

from statsmodels.stats.anova import AnovaRM # perform F test
# setting working directory 
wkdir = '/Volumes/diedrichsen_data$/data/Cerebellum/CerebellumWorkingMemory/selective_recruit'

def plot():
    """
    """
    # load the dataframe for the whole 
    df_path = os.path.join(wkdir, "ROI_all.tsv")
    df = pd.read_csv(df_path, sep="\t")

    label_dict={1:'Enc2F', 2:'Ret2F',
                3:'Enc2B', 4:'Ret2B',
                5:'Enc4F', 6:'Ret4F',
                7:'Enc4B', 8:'Ret4B',
                9:'Enc6F', 10:'Ret6F',
                11:'Enc6B',12:'Ret6B',
                13:'rest'}
    marker_dict = {1:'o',2:'X',
                   3:'o',4:'X',
                   5:'o',6:'X',
                   7:'o',8:'X',
                   9:'o',10:'X',
                   11:'o',12:'X',
                   13:'s'}
    color_dict  = {1:'b',2:'b',
                   3:'r',4:'r',
                   5:'b',6:'b',
                   7:'r',8:'r',
                   9:'b',10:'b',
                   11:'r',12:'r',
                   13:'g'}

    #   prepare df for plotting
    #   by setting non-numeric fields to 'first', we make sure that they are not removed from the final dataframe
    plotting.make_scatterplot2(df,split='cond_num',
            labels=label_dict,
            markers=marker_dict,
            colors=color_dict)
    

if __name__ == "__main__":
    plot()
    pass