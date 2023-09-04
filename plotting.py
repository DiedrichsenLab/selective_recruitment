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
# from nilearn import plotting
import SUITPy.flatmap as flatmap
import surfAnalysisPy as sa


from scipy import stats as sps # to calcualte confidence intervals, etc
from statsmodels.stats.anova import AnovaRM # perform F test

import selective_recruitment.globals as gl
import selective_recruitment.region as sroi

import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am

import cortico_cereb_connectivity.scripts.script_summarize_weights as wplot

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

   # adjust_text(texts) # make sure you have installed adjust_text

def make_scatterplot(dataframe, 
                     x = "X", 
                     y = "Y", 
                     split='cond_num', 
                     fit_line = True, 
                     labels=None,
                     colors=None,
                     markers='o'):
    """
    make scatterplot
    Args:
    dataframe (pd.DataFrame) -
            entire dataframe with individual subject data and fitted slopes and intercepts
    split (str) - column name indicating the different conditions to be plotted
    fit_line (bool) - whether to fit a line to the data
    labels (str) - column name indicating the labels to be used for the data points
    """
    # do the scatter plot
    grouped = dataframe.groupby([split])
    agg_kw = {split:'first',
              x:np.mean,
              y: np.mean,
             'slope':np.mean,
             'intercept':np.mean}
    if isinstance(labels,str):
        agg_kw[labels] = 'first'
    df = grouped.agg(agg_kw)
    if isinstance(labels,(list,np.ndarray)):
        df['labels']=labels
        labels = 'labels'
    elif isinstance(labels,(pd.Series,dict)):
        df['labels']=df[split].map(labels)
        labels = 'labels'

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
            labels = df[labels])
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

def plot_parcels(parcellation = "NettekovenSym32",
                 atlas_space = "SUIT3", 
                 roi_exp = "D.?R", 
                 split = None, 
                 stats = "mode", 
                 render = "plotly", 
                 cmap = 'tab20b'):
    
    # make an atlas object
    atlas, ainfo = am.get_atlas(atlas_str = atlas_space, atlas_dir = gl.atlas_dir)    
    
    # get the list of all the regions
    label_name_list = sroi.get_parcel_names(parcellation = parcellation, 
                                           atlas_space = atlas_space)
    
    # get the mask and names of the selected regions
    mask, idx, selected_ = sroi.get_parcel_single(parcellation = parcellation, 
                                              atlas_space = atlas_space,
                                              roi_exp = roi_exp)
    
    print(selected_)
        
    # load the label file
    fname = gl.atlas_dir + f'/{ainfo["dir"]}/atl-{parcellation}_space-SUIT_dseg.nii'
    img = nb.load(fname)   
    # use get parcel to get list of labels 
    label_vec, labels = atlas.get_parcel(fname)
    
    # make a nifti image for the selected regions
    dat_new = np.zeros_like(label_vec, dtype = float)
    if split is None: # merge into one single parcel
        dat_new[np.isin(label_vec, idx)] = 1
    else:
        for i, r in enumerate(idx):
            dat_new[np.isin(label_vec, r)] = split[i]

    
    img = atlas.data_to_nifti(dat_new)
    
    # map it from volume to surface
    img_flat = flatmap.vol_to_surf([img], stats=stats, space = 'SUIT', ignore_zeros=True)
    ax = flatmap.plot(img_flat, render=render, bordersize = 1.5, 
                      overlay_type='label', cmap = cmap)
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
                             cscale = [-0.0001, 0.0001],
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
                                        vmax = cscale[1],
                                    )
        print(np.nanmax(weight_roi_list[0]))
        ax.append(fig.figure)
    return ax

def plot_significant_activity_cortex(data, 
                                        cortex_roi = "Icosahedron1002", 
                                        dataset = "WMFS", 
                                        
                                        ):
    # create an instance of the dataset class
    Data = ds.get_dataset_class(base_dir = gl.base_dir, dataset = "WMFS")
    ## make atlas object first
    atlas_fs, _ = am.get_atlas("fs32k", gl.atlas_dir)
    
    # load the label file for the cortex
    label_fs = [gl.atlas_dir + f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]

    # get parcels for the neocortex
    _, label_fs = atlas_fs.get_parcel(label_fs, unite_struct = False)
    
    # get the maps for left and right hemispheres
    surf_map = []
    for label in atlas_fs.label_list:
        # loop over regions within the hemisphere
        label_arr = np.zeros([data.T.shape[0], label.shape[0]])
        for p in np.arange(1, data.T.shape[0]):
            for i in np.unique(label):            
                np.put_along_axis(label_arr[p-1, :], np.where(label==i)[0], data.T[p-1,i-1], axis=0)
        surf_map.append(label_arr)

    cifti_img = atlas_fs.data_to_cifti(surf_map)
    
    return cifti_img
    


def plot_rgb_map(data_rgb, 
                 atlas_space = "SUIT3", 
                 render = "plotly", 
                 scale = [0.02, 1, 0.02], 
                 threshold = [0.02, 1, 0.02]):
    """
    plots rgb map of overlap on flatmap
    Args:
        data_rgb (np.ndarray) - 3*p array containinig rgb values per voxel/vertex
        atlas_space (str) - the atlas you are in, either SUIT3 or fs32k
        scale (list) - how much do you want to scale
        threshold (list) - threshold to be applied to the values
    Returns:
        ax (plt axes object) 
    """
    atlas, a_info = am.get_atlas(atlas_str=atlas_space, atlas_dir=gl.atlas_dir)
    if atlas_space == "SUIT3":
        Nii = atlas.data_to_nifti(data_rgb)
        data = flatmap.vol_to_surf(Nii,space='SUIT')
        rgb = flatmap.map_to_rgb(data,scale=scale,threshold=threshold)
        ax = flatmap.plot(rgb,overlay_type='rgb', colorbar = True, render = render)
    elif atlas_space == "fs32k":
        dat_cifti = atlas.data_to_cifti(data_rgb)
        # get the lists of data for each hemi
        dat_list = nt.surf_from_cifti(dat_cifti)
        ax = []
        for i,hemi in enumerate(['L', 'R']):
            plt.figure()
            rgb = flatmap.map_to_rgb(dat_list[i].T,scale,threshold=threshold, render = "matplotlib")
            ax.append(sa.plot.plotmap(rgb, surf = f'fs32k_{hemi}',overlay_type='rgb'))

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
    # ax.plot(x, y, color = 'black')
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
    return ax

def plot_mds_sns(data, x, y, label = "roi_hemi", hue = 'roi_idx', style = 'roi_ap', palette = "Dark2", vectors = None,v_labels = None):
    ax = plt.gca()
    # Scatter plot with axis equal
    sns.scatterplot(data=data,x=x,y=y,hue=hue,style=style, palette=palette,s=100, ax=ax)
    
    texts = []
    for i,l in enumerate(data[label]):
        text = ax.text(
                        data[x][i] + 0.001,
                        data[y][i],
                        s = l,
                        horizontalalignment='left',
                        size="medium",
                        weight='regular'
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
    return ax


def plot_mds3(x, y, z, label, colors=None,text_size = 'small', text_weight = 'regular',vectors = None,v_labels = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax = plt.gca(projection='3d')
    ax.scatter(x,y, z,s=70,c=colors)
        
    
    # ax.plot(x, y, z, color='black')
    ax.set_box_aspect((1, 1, 1))
    
    # set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
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

def plot_mds3_plotly(x, y,z, 
                  vectors = None,
                  label = "NettekovenSym68c32", 
                  roi_super = "D", 
                  hue = "roi_super", 
                  text = "roi_name", 
                  vec_labels = ['retrieval+','load+','backwards+']):
    # get region info
    Dinfo,D_indx, colors_D = sroi.get_parcel_names(parcellation = label, atlas_space = "SUIT3")
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
    plot_connectivity_weight(roi_name = "D2R",
                             method = "L2Regression",
                             cortex_roi = "Icosahedron1002",
                             cerebellum_roi = "NettekovenSym32",
                             cerebellum_atlas = "SUIT3",
                             log_alpha = 8,
                             view = "lateral", 
                             dataset_name = "MDTB",
                             cmap = "coolwarm",
                             colorbar = True, 
                             cscale = [-0.0001, 0.0001],
                             ses_id = "ses-s1")
    pass