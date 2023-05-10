import os
import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns # for plots
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text # to adjust the text labels in the plots (pip install adjustText)

import nibabel as nb
import nitools as nt
from nilearn import plotting
from SUITPy import flatmap

from scipy import stats as sps # to calcualte confidence intervals, etc
from statsmodels.stats.anova import AnovaRM # perform F test

import selective_recruitment.globals as gl
import selective_recruitment.region as sroi

import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am

import cortico_cereb_connectivity.scripts.script_plot_weights as wplot

# making the scatterplot
def annotate(dataframe, x = "X", y = "Y", labels = 'cond_num', text_size = 'small', text_weight = 'regular'):
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
                        d[x]+0.001,
                        d[y],
                        s = labels.loc[i],
                        horizontalalignment='left',
                        size=text_size,
                        weight=text_weight
                        )
        texts.append(text)

    adjust_text(texts) # make sure you have installed adjust_text

def make_scatterplot(dataframe, x = "X", y = "Y", split='cond_num', fit_line = True, labels=None,
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
              x:np.mean,y: np.mean,
             'slope':np.mean,
             'intercept':np.mean}
    df = grouped.agg(agg_kw)

    df["Y_CI"] = grouped.Y.apply(sps.sem) * 1.96
    df["X_CI"] = grouped.X.apply(sps.sem)*1.96
    df['X_err'] = grouped.res.apply(sps.sem)*1.96

    # add  the appropriate errorbars
    plt.errorbar(x = df[x],
                 y = df[y],
                 yerr = df['X_err'],
                 elinewidth=2,
                fmt='none', # no marker will be used when plotting the error bars
                color=(0.3,0.3,0.3),
                ecolor=(0.5,0.5,0.5)
                )

    # Plot average regression line
    if fit_line:
        xrange = np.array([df[x].min(),df[x].max()])
        ypred = xrange*df.slope.mean()+df.intercept.mean()
        plt.plot(xrange,ypred,'k-')

    # Make scatterplot, determining the markers and colors from the dictionary
    ax = sns.scatterplot(data=df, x='X', y='Y', style = split, hue = split, s = 100,legend=None,markers=markers,palette=colors)

    # set labels
    ax.set_xlabel('Cortical Activation (a.u.)')
    ax.set_ylabel('Cerebellar Activation (a.u.)')

    # get labels for each data point
    annotate(df, x = x, y = y,
            text_size = 'small',
            text_weight = 'regular',
            labels = df[split].map(labels))
    return

# plotting cortical and cerebellar maps
def plot_activation_map(dataset = "WMFS", 
                         ses_id = "ses-02", 
                         subj = "group",
                         type = "CondAll", 
                         atlas_space = "SUIT3", 
                         contrast_name = "average", 
                         render = "plotly", 
                         view = "lateral", 
                         cmap = "coolwarm",
                         cscale = [-0.2, 0.2], 
                         colorbar = True, 
                         smooth = None):
    """
    """
    # get dataset
    data,info,dset = ds.get_dataset(gl.base_dir,
                                    dataset = dataset,
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type,  
                                    smooth = smooth)

    # get the contrast of interest
    if contrast_name == "average":
        # make up a numpy array with all 1s as we want to average all the conditions
        idx = np.ones([len(info.index), ], dtype = bool)
    else:
        idx = (info.names == contrast_name).values

    # get the data for the contrast of interest
    dat_con = np.nanmean(data[0, idx, :], axis = 0)

    # prepare data for plotting
    atlas, ainfo = am.get_atlas(atlas_space, gl.atlas_dir)
    if atlas_space == "SUIT3":
        # convert vol 2 surf
        img_nii = atlas.data_to_nifti(dat_con)
        # convert to flatmap
        img_flat = flatmap.vol_to_surf([img_nii], 
                                        stats='nanmean', 
                                        space = 'SUIT', 
                                        ignore_zeros=True)
        ax = flatmap.plot(data=img_flat, 
                          render=render, 
                          hover='auto', 
                          cmap = cmap, 
                          colorbar = colorbar, 
                          bordersize = 1, 
                          cscale = cscale)

    elif atlas_space == "fs32k":
        # get inflated cortical surfaces
        surfs = [gl.atlas_dir + f'/tpl-fs32k/tpl_fs32k_hemi-{h}_inflated.surf.gii' for i, h in enumerate(['L', 'R'])]

        # first convert to cifti
        img_cii = atlas.data_to_cifti(dat_con.reshape(-1, 1).T)
        img_con = nt.surf_from_cifti(img_cii)
        
        ax = []
        for h,hemi in enumerate(['left', 'right']):
            fig = plotting.plot_surf_stat_map(
                                            surfs[h], img_con[h], hemi=hemi,
                                            # title='Surface left hemisphere',
                                            colorbar=colorbar, 
                                            view = view,
                                            cmap=cmap,
                                            engine='plotly',
                                            symmetric_cbar = True,
                                            vmax = cscale[1]
                                        )

            ax.append(fig.figure)
    return ax

def plot_mapwise_recruitment(data, 
                            atlas_space = "SUIT3",  
                            render = "plotly", 
                            cmap = "hsv", 
                            cscale = [-5, 5], 
                            threshold = None):
    """
    plots results of the map-wise selective recruitment on the flatmap
    """
    if threshold is not None:
        # set values outside threshold to nan
        data[np.abs(data)<=threshold] = np.nan

    atlas,ainf = am.get_atlas(atlas_space, gl.atlas_dir)
    X = atlas.data_to_nifti(data)
    sdata = flatmap.vol_to_surf(X)
    fig = flatmap.plot(sdata, render=render, colorbar = True, cmap = cmap, cscale = cscale, bordersize = 1.5)
    return fig

def plot_parcels_super(label = "NettekovenSym68c32", 
                       roi_super = "D", 
                       render = "plotly"):
    """
    Plots the selected super region based on the parcellation defined by "parcellation"
    Args:
        parcellation (str) - name of the hierarchical parcellation 
        roi_super (str) - name assigned to the roi in the hierarchical parcellation
    Returns:
        None
    """
    # get the roi numbers for the super roi
    idx_label, colors2, label_names = nt.read_lut(f"{gl.atlas_dir}/tpl-SUIT/atl-{label}.lut")
    
    D_indx = [label_names.index(name) for name in label_names if roi_super in name]
    D_name = [name for name in label_names if roi_super in name]
    D_name.insert(0, '0')

    fname = gl.atlas_dir + f'/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii'
    img = nb.load(fname)
    dat = img.get_fdata()
    
    # map it from volume to surface
    img_flat = flatmap.vol_to_surf([img], stats='mode', space = 'SUIT')

    # get a mask for the selected super region
    region_mask = np.isin(img_flat.astype(int), D_indx)

    # convert non-selected labels to nan
    roi_flat = img_flat.copy()
    # convert non-selected labels to nan
    roi_flat[np.logical_not(region_mask)] = np.nan

    for i, r in enumerate(D_indx):
        roi_flat[roi_flat == r] = i+1
    
    ax = flatmap.plot(roi_flat, render=render, bordersize = 1.5, 
                      overlay_type='label',
                      label_names=D_name, cmap = 'tab20b')
    return ax

def plot_parcels_single(label = "NettekovenSym68c32", 
                        roi_name = "D1R", 
                        render = "plotly"):
    """
    plot the selected region from parcellation on flatmap
    Args:
        parcellation (str) - name of the parcellation
        roi_name (str) - name of the roi as stored in the lookup table
    Return:
        ax (axes object)
        roi_num (int) - number corresponding to the region
    """
    fname = gl.atlas_dir + f'/tpl-SUIT/atl-{label}_space-SUIT_dseg.nii'
    img = nb.load(fname)
    # map it from volume to surface
    img_flat = flatmap.vol_to_surf([img], stats='mode', space = 'SUIT', ignore_zeros=True)

    # get the lookuptable for the parcellation
    lookuptable = nt.read_lut(gl.atlas_dir + f'/tpl-SUIT/atl-{label}.lut')

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
    ax = flatmap.plot(roi_flat, render=render,
                      hover='auto', colorbar = False,
                      bordersize = 1.5, overlay_type='label',
                      label_names=label_info, cmap = cmap)

    return ax

def plot_connectivity_weight(roi_name = "D2R",
                             method = "L2Regression",
                             cortex_roi = "Icosahedron1002",
                             cerebellum_roi = "NettekovenSym68c32",
                             cerebellum_atlas = "SUIT3",
                             log_alpha = 8,
                             view = "lateral", 
                             dataset_name = "MDTB",
                             cmap = "coolwarm",
                             colorbar = True, 
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

    surfs = [gl.atlas_dir + f"/tpl-fs32k/tpl_fs32k_hemi-{hemi}_inflated.surf.gii" for hemi in ['L', 'R']]
    ax = []
    for h, hemi in enumerate(['left', 'right']):

        fig = plotting.plot_surf_stat_map(
                                        surfs[h], weight_roi_list[h], hemi=hemi,
                                        # title='Surface left hemisphere',
                                        colorbar=colorbar, 
                                        view = view,
                                        cmap=cmap,
                                        engine='plotly',
                                        symmetric_cbar = True,
                                        vmax = np.nanmax(weight_roi_list[0]),
                                    )
        print(np.nanmax(weight_roi_list[0]))
        ax.append(fig.figure)
    return ax

# MDS plots
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
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax = plt.gca(projection='3d')
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

def plot_mds3_new(x, y,z, 
                  vectors = None,
                  label = "NettekovenSym68c32", 
                  roi_super = "D", 
                  hue = "roi_super", 
                  text = "roi_name", 
                  vec_labels = ['retrieval+','load+','backwards+']):
    # get region info
    Dinfo,D_indx, colors_D = sroi.get_region_info(label = label, roi_super = roi_super)
    # adding the components to the D region info dataframe
    Dinfo["comp_0"] = x
    Dinfo["comp_1"] = y
    Dinfo["comp_2"] = z
    # add index
    Dinfo["roi_idx"] = Dinfo["roi_name"].str[1].astype(int)
    Dinfo["roi_super"] = Dinfo["roi_name"].str[0]

    fig = px.scatter_3d(Dinfo, x="comp_0", y="comp_1", z="comp_2", text = text, color = hue)
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    if vectors is not None:
        
        x_min = min(x)
        x_max = max(x)

        # scale the vectors
        scl=(x_max-x_min)/4
        vec = vectors*scl

        # make vectors
        m, n = vec.shape        
        lines = np.zeros((m,2*n),dtype=vec.dtype)
        lines[:,::-2] = vec
        trace1 = go.Scatter3d(
            x=lines[0, :],
            y=lines[1, :],
            z=lines[2, :],
            mode='lines+text',
            text=vec_labels,
            textposition=['top center', 'middle center', 'bottom center'],
            name='contrasts', 
            
        )
        fig.add_trace(trace1)
    fig.show()
    return 


if __name__ == "__main__":
    pass