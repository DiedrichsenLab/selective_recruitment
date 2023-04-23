import os
import numpy as np
import seaborn as sns # for plots
import nibabel as nb
import nitools as nt
from nilearn import plotting
import selective_recruitment.globals as gl
import Functional_Fusion.dataset as fdata
import Functional_Fusion.atlas_map as am
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pathlib import Path
from SUITPy import flatmap
import matplotlib.pyplot as plt
from scipy import stats as sps # to calcualte confidence intervals, etc
from adjustText import adjust_text # to adjust the text labels in the plots (pip install adjustText)

import cortico_cereb_connectivity.scripts.script_plot_weights as wplot

from statsmodels.stats.anova import AnovaRM # perform F test

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

def get_label_info(parcellation):
    # get the lookuptable for the parcellation
    lookuptable = nt.read_lut(gl.atlas_dir + f'/tpl-SUIT/atl-{parcellation}.lut')

    # get the label info
    label_info = lookuptable[2]
    if '0' not in label_info:
        # append a 0 to it
        label_info.insert(0, '0')
    cmap = LinearSegmentedColormap.from_list("color_list", lookuptable[1])
    return label_info

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

def plot_cerebellum_activation(dataset, ses_id, subject, contrast_name):
    """_summary_
    Args:
        subject (_type_): _description_
        contrast_name (_type_): _description_
    Returns:
        _type_: _description_
    """

    # create an instance of WMFS class
    Dataset = fdata.get_dataset_class(gl.base_dir, dataset= dataset)
    # get info
    info = Dataset.get_info(ses_id,"CondAll")

    # get participants
    subjects = list(Dataset.get_participants().participant_id.values)
    subjects.append("group")

    path_to_cifti = Dataset.data_dir.format(subject) + f'/{subject}_space-SUIT3_{ses_id}_CondAll.dscalar.nii'
    cifti_img = nb.load(path_to_cifti)

    # get the numpy array
    img = cifti_img.get_fdata()
    # get info for the contrast
    info_con = info.loc[info.names == contrast_name]
    idx = info_con.reg_id.values[0]

    # get the numpy array corresponding to the contrast
    img_con = img[idx-1, :].reshape(-1, 1)

    # convert vol 2 surf
    atlas_cereb, ainfo = am.get_atlas('SUIT3',gl.atlas_dir)
    img_nii = atlas_cereb.data_to_nifti(img_con.T)
    img_flat = flatmap.vol_to_surf([img_nii], stats='nanmean', space = 'SUIT')

    # plot
    ax = flatmap.plot(data=img_flat, render="plotly", cmap = "coolwarm", colorbar = True, bordersize = 1.5, cscale=[-0.2, 0.2])
    # ax.show()
    return ax

def plot_cortex_activation(dataset, ses_id, subject, contrast_name):
    """_summary_
    Args:
        subjcet (_type_): _description_
        contrast_name (_type_): _description_
    """
    # get surfaces
    # surfs = [gl.atlas_dir + f"/tpl-fs32k/tpl_fs32k_hemi-{hemi}_inflated.surf.gii" ]

    # create an instance of WMFS class
    Dataset = fdata.get_dataset_class(gl.base_dir, dataset= dataset)
    # get info
    info = Dataset.get_info(ses_id,"CondAll")

    # get participants
    subjects = list(Dataset.get_participants().participant_id.values)
    subjects.append("group")

    path_to_cifti = Dataset.data_dir.format(subject) + f'/{subject}_space-fs32k_{ses_id}_CondAll.dscalar.nii'
    cifti_img = nb.load(path_to_cifti)

    # get the hemispheres
    cifti_list = nt.surf_from_cifti(cifti_img)

    # get info for the contrast
    info_con = info.loc[info.names == contrast_name]
    idx = info_con.reg_id.values[0]
    surf_hemi = []
    fig_hemi = []
    for h, hemi in enumerate(['L', 'R']):
        img = cifti_list[h]
        # get the numpy array corresponding to the contrast
        img_con = img[idx-1, :].reshape(-1, 1)

        surf_hemi.append(gl.atlas_dir + f"/tpl-fs32k/tpl_fs32k_hemi-{hemi}_inflated.surf.gii")

        fig_hemi.append(plotting.view_surf(
                                        surf_hemi[h], img_con, colorbar=True,
                                        threshold=None, cmap="coolwarm" , vmax = 1
                                        ))
    return fig_hemi

def plot_connectivity_weight(roi_name = "D2R",
                             method = "L2Regression",
                             cortex_roi = "Icosahedron1002",
                             cerebellum_roi = "NettekovenSym68c32",
                             cerebellum_atlas = "SUIT3",
                             log_alpha = 8,
                             dataset_name = "MDTB",
                             cmap = "coolwarm",
                             ses_id = "ses-s1"):
    """
    """
    # get connectivity weight maps for the selected region
    cifti_img = wplot.get_weight_map(method = method,
                                    cortex_roi = cortex_roi,
                                    cerebellum_roi = cerebellum_roi,
                                    cerebellum_atlas = cerebellum_atlas,
                                    log_alpha = log_alpha,
                                    dataset_name = dataset_name,
                                    ses_id = ses_id,
                                    type = "dscalar"
                                    )

    # get the cortical weight map corresponding to the current
    ## get parcel axis from the cifti image
    parcel_axis = cifti_img.header.get_axis(0)
    ## get the name of the parcels in the parcel_axis
    idx = list(parcel_axis.name).index(roi_name)
    # get the maps for left and right hemi
    weight_map_list = nt.surf_from_cifti(cifti_img)
    # get the map for the selected region for left and right hemispheres
    weight_roi_list = [weight_map_list[h][idx, :] for h in [0, 1]]

    surf_hemi = []
    fig_hemi = []
    for h, hemi in enumerate(['L', 'R']):
        img = weight_map_list[h]
        # get the numpy array corresponding to the contrast
        img_data = weight_roi_list[h]
        surf_hemi.append(gl.atlas_dir + f"/tpl-fs32k/tpl_fs32k_hemi-{hemi}_inflated.surf.gii")

        fig_hemi.append(plotting.view_surf(
                                        surf_hemi[h], img_data, colorbar=True,
                                        cmap=cmap, vmax = np.nanmax(img_data),
                                        vmin = np.nanmin(img_data)
                                        ))
    return fig_hemi

def roi_difference(df,
             xvar = "cond_name",
             hue = "roi_name",
             depvar = "Y",
             sub_roi = None,
             roi = "D",
             var = ["cond_name", "roi_name"]):
    """ roi_difference plots and tests for differences between rois
    """
    # get D regions alone?
    names = df.roi_name.values.astype(str)
    mask_roi = np.char.startswith(names, roi)

    # add a new column determining side (hemisphere)
    df["side"] = df["roi_name"].str[2]

    # add a column determining anterior posterior
    mask_anteriors = np.char.endswith(names, "A")
    df["AP"] = ""
    df["AP"].loc[mask_anteriors] = "A"
    df["AP"].loc[np.logical_not(mask_anteriors)] = "P"

    # add a new column that defines the index assigned to the region
    # for D2 it will be 2, for D3 it will be 3
    df["sub_roi_index"] = df["roi_name"].str[1]

    # add a new column determining side (hemisphere)
    df["side"] = df["roi_name"].str[2]

    # get Ds
    DD_D = df.loc[(mask_roi)]
    # get the specific region
    if sub_roi is not None:
        DD_D = DD_D.loc[DD_D.sub_roi_index == sub_roi]

    # barplots
    plt.figure()
    ax = sns.lineplot(data=DD_D.loc[(df.cond_name != "rest")], x = xvar, y = depvar,
                    errwidth=0.5, hue = hue)
    plt.xticks(rotation = 90)
    anov = AnovaRM(data=df, depvar=depvar,
                  subject='sn', within=var, aggregate_func=np.mean).fit()
    return anov

def calc_mds(X,center=True,K=2):
    """
    calculate MDS for the given matrix
    Args:
    X (np.array) - matrix for which MDS is to be calculated
    Returns:
    W (np.array) - MDS coordinates
    V (np.array) - direction in column space
    """
    if center:
        X=X-X.mean(axis=0)
    W,S,V=np.linalg.svd(X,full_matrices=False)
    W = W[:,:K]
    S = S[:K]
    V = V[:K,:] # V is already transposed
    return W*S,V


def plot_mds(x, y, label, colors=None,text_size = 'small', text_weight = 'regular',vectors = None,v_labels = None):
    ax = plt.gca()
    # Scatter plot with axis equal
    ax.scatter(x, y, s=100, c = colors)
    ax.axis('equal')
    texts = []
    for i,l in enumerate(label):
        text = ax.text(
                        x[i] + 0.001,
                        y[i],
                        s = l,
                        horizontalalignment='left',
                        size=text_size,
                        weight=text_weight
                        )
        texts.append(text)
    adjust_text(texts) # make sure you have installed adjust_text
    if vectors is not None:
        scl=(ax.get_xlim()[1]-ax.get_xlim()[0])/4
        v = vectors*scl

        for i in range(vectors.shape[1]):
            ax.quiver(0,0,v[0,i],v[1,i],angles='xy',scale_units='xy',width=0.002,scale=1.0)
            if v_labels is not None:
                ax.text(v[0,i]*1.05,v[1,i]*1.05,v_labels[i],horizontalalignment='center',verticalalignment='center')
    return

def plot_mds3(x, y, z, label, colors=None,text_size = 'small', text_weight = 'regular',vectors = None,v_labels = None):
    ax = plt.gca(projection='3d')
    ax.scatter(x,y, z,s=70,c=colors)
    ax.set_box_aspect((1, 1, 1))
    texts = []

    for i,l in enumerate(label):
        text = ax.text(
                    x[i] + 0.001,
                    y[i],
                    z[i],
                    l,
                    horizontalalignment='left',
                    size=text_size,
                    weight=text_weight
                    )
        texts.append(text)
    adjust_text(texts) # make sure you have installed adjust_text
    if vectors is not None:
        scl=(ax.get_xlim()[1]-ax.get_xlim()[0])/4
        v = vectors*scl
        for i in range(vectors.shape[1]):
            ax.quiver(0,0,0,v[0,i],v[1,i],v[2,i],normalize=False)
            if v_labels is not None:
                ax.text(v[0,i]*1.05,v[1,i]*1.05,v[2,i]*1.05,v_labels[i],horizontalalignment='center',verticalalignment='center')
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

    if xerr is not None:
        df['xerr'] = grouped[xerr].apply(sps.sem)*1.96
    else:
        df['xerr'] = None
    if yerr is not None:
        df['yerr'] = grouped[yerr].apply(sps.sem)*1.96
    else:
        df['yerr'] = None

    # add  the appropriate errorbars
    plt.errorbar(x = df[xlabel],
                 y = df[ylabel],
                #  yerr = df['xerr'],
                #  xerr = df['yerr'],
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