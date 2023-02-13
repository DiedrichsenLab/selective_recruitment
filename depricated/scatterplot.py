import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from pathlib import Path
import numpy as np

base_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/Language'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/Cerebellum/Language'
if not Path(base_dir).exists():
    base_dir = '//bmisrv.robarts.ca/Diedrichsen_data$/data/Cerebellum/Language'
if not Path(base_dir).exists():
    base_dir = '/cifs/diedrichsen/data/Cerebellum/Language'

def mean_scatterplot(dest_file):
    d = pd.read_csv(f'{base_dir}/{dest_file}')
    g = d.groupby(["CN"]).mean()
    ax = sns.scatterplot(data=g, x = "cortex", y = "cerebellum",hue='CN', s= 50,style='CN')
    m, b = np.polyfit(g.cortex, g.cerebellum, 1)
    ax.plot(g.cortex, m*g.cortex+b, color = "gray", linewidth = 0.5)
    X_sem=d.groupby('CN')['cortex'].sem()
    Y_sem=d.groupby('CN')['cerebellum'].sem()
    plt.errorbar(data=g, x='cortex', y='cerebellum', yerr=Y_sem,linestyle='None')


def indi_scatterplot(dest_file):
    d = pd.read_csv(f'{base_dir}/{dest_file}')
    ax = sns.scatterplot(data=d, x="cortex", y="cerebellum",
                         hue='CN', s=50, style='CN')
    m, b = np.polyfit(d.cortex, d.cerebellum, 1)
    ax.plot(d.cortex, m * d.cortex + b, color="gray", linewidth=0.5)

if __name__ == "__main__":
    # debug individual scatter plot
    # indi_scatterplot('language_data_analyses/across_whole_spoken/spoken.csv')
    pass
