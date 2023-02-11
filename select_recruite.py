#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions used to do selective recruitment analysis

Created on 12/09/2022 at 5:25 PM
Author: Ladan Shahshahani
"""
# import packages
import sys

import numpy as np
import pandas as pd
import deepdish as dd
import seaborn as sb
from pathlib import Path

# modules from functional fusion
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.matrix as matrix

# modules from connectivity
import cortico_cereb_connectivity.prepare_data as cprep

import os
import nibabel as nb
import nitools as nt

# visualization tools
import SUITPy as suit
from surfplot import Plot as plot_surf
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import rsatoolbox as rst
# import rsatoolbox.data as rsd

# set base directory of the functional fusion 
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'


# Get smoothing matrix, can be used to smooth the weights (for connectivity)
def get_smooth_matrix(atlas, fwhm = 3):
    """
    calculates the smoothing matrix to be applied to data
    Args:
        atlas - experiment name (fs or wm)
        fwhm (float)  - fwhm of smoothing kernel
    Rerurns:
        smooth_mat (np.ndarray) - smoothing matrix
    """
    # get voxel coordinates
    
    
    # calculate euclidean distance
    euc_dist = nt.euclidean_dist_sq(atlas.vox, atlas.vox)

    smooth_mat = np.exp(-1/2 * euc_dist/(fwhm**2))
    smooth_mat = smooth_mat /np.sum(smooth_mat, axis = 1);   

    return smooth_mat

def calc_mean(data,info,
                partition='run',
                condition='reg_id',
                reorder=False):
    n_subj = data.shape[0]
    cond=np.unique(info.reg_id)
    n_cond = len(cond)

    # For this purpose, ignore nan voxels
    data = np.nan_to_num(data,copy=False)
    
    mean_d = data.mean(axis=2)
    Z = matrix.indicator(info.reg_id)
    mean_d = mean_d @ np.linalg.pinv(Z).T
    
    part = np.unique(info[partition])
    inf=info[info[partition]==part[0]].copy()
    if reorder:
        inf=inf.sort_values(reorder)
        ind=inf.index.to_numpy()
        inf=inf.reset_index()
        mean_d = mean_d[:,ind]
    
    return mean_d,inf




# use connectivity model to predict cerebellar activation
def predict_cerebellum(weights, scale, X, atlas, info, fwhm = 0):
    """
    makes predictions for the cerebellar activation
    uses weights from a linear model (w) and cortical data (X)
    to make predictions Yhat
    Args:
    X (np.ndarray)      - cortical data
    weights (np.ndarray) - connectivity weights
    scale (np.ndarray) - used to scale data
    atlas (atlas object) - atlas object (will be used in smoothing)
    info (pd.DataFrame) - pandas dataframe representing task info
    fwhm (int) - smoothing
    Returns:
    Yhat (np.ndarray) - predicted cerebellar data
    """
    # apply scaling
    X = X / scale

    # get smoothing matrix 
    if fwhm != 0:
        smooth_mat = get_smooth_matrix(atlas, fwhm)
        weights = smooth_mat@weights

    # make predictions
    Yhat = np.dot(X, weights.T)
    Yhat = np.r_[Yhat[info.half == 2, :], Yhat[info.half == 1, :]]
    return Yhat

# regress cerebellar data onto cortical/cerebellar predictions
def regressXY(X, Y, fit_intercept = False):
    """
    regresses Y onto X.
    Will be used to regress observed cerebellar data onto predicted
    Args:
        X (np.ndarray) - predicted cerebellar data for each roi
        Y (np.ndarray) - observed cerebellar data for each roi
        subtract_mean (boolean) - subtract mean before regression?
    Returns:
        coef (np.ndarray) - regression coefficients
        residual (np.ndarray) - residuals 
        R2 (float) - R2 of the regression fit
    """

    # Estimate regression coefficients
    # X = X.reshape(-1, 1)
    # Y = Y.reshape(-1, 1)
    if fit_intercept:
        X = np.c_[ np.ones(X.shape[0]), X ]  
    # coef = np.linalg.inv(X.T@X) @ (X.T@Y)
    
    # # matrix-wise simple regression? NOT USED HERE
    # # c = np.sum(X*Y, axis = 0) / np.sum(X*X, axis = 0)

    # # Calculate residuals
    # residual = Y - X@coef
    # print(sum(residual))


    model = np.polyfit(X, Y, 1)
    predict = np.poly1d(model)
    residual = Y - predict(X)
    # print(sum(residual))

    # calculate R2 
    rss = sum(residual**2)
    tss = sum((Y - np.mean(Y))**2)
    R2 = 1 - rss/tss

    return model[1], residual, R2

def run_regress(X,Y,info,fit_intercept = False):
    # Looping over subject and running the regression for 
    # Each of them. 
    n_subj,n_cond = X.shape
    summary_list = [] 
    for i in range(n_subj):
        info_sub = info.copy()
        x = X[i,:]
        y = Y[i,:]

        coef, res, R2 = regressXY(x, y, fit_intercept = fit_intercept)
        info_sub["sn"]    = i * np.ones([len(info_sub), 1])
        info_sub["X"]     = x # X is either the cortical data or the predicted cerebellar activation
        info_sub["Y"]     = y
        info_sub["res"]   = res
        info_sub["coef"]  = coef * np.ones([len(info_sub), 1])
        info_sub["R2"]    = R2 * np.ones([len(info_sub), 1])

        summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0)
    return summary_df

# getting data into a dataframe
def get_summary(outpath = None,
                     dataset_name = "WMFS",
                     cerebellum = "wm_verbal", 
                     cortex = 'Icosahedron-1002.32k', 
                     predict = True, 
                     conn_dataset = 'MDTB',
                     conn_ses_id  = 'ses-s1',
                     conn_method = "L2Regression", 
                     log_alpha = 8, 
                     ses_id = 'ses-02',
                     type = "CondHalf",
                     unite_struct = False, 
                     save_tensor = False):
    """
    prepares a dataframe for plotting the scatterplot
    """

    # get dataset class object
    Data = get_dataset_class(base_dir, dataset=dataset_name)

    # get info
    info = Data.get_info(ses_id,type)

    # get list of subjects:
    T = Data.get_participants()

    if save_tensor:
        # get data tensor for SUIT3
        cdata.save_data_tensor(dataset = dataset_name,
                        atlas='SUIT3',
                        sess=ses_id,
                        type=type)

        # get data tensor for fs32k
        cdata.save_data_tensor(dataset = dataset_name,
                        atlas='fs32k',
                        sess=ses_id,
                        type=type)



    # load data tensor for SUIT3 (TODO: add an option to specify atlases other than SUIT3 and fs32k)
    file_suit = outpath + f'/{dataset_name}/{dataset_name}_SUIT3_{ses_id}_{type}.npy'
    cdat = np.load(file_suit)

    # load data tensor for fs32k
    file_fs32k = outpath + f'/{dataset_name}/{dataset_name}_fs32k_{ses_id}_{type}.npy'
    ccdat = np.load(file_fs32k)
    

    # create instances of atlases for the cerebellum and cortex
    atlas_cereb, ainfo = am.get_atlas('SUIT3',atlas_dir)
    atlas_cortex, ainfo = am.get_atlas('fs32k', atlas_dir)

    # get label files for cerebellum and cortex
    # NOTE: To average over cerebellum or cortex, pass on masks as label files
    label_cereb = atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum}_space-SUIT_dseg.nii'
    label_cortex = []
    for hemi in ['L', 'R']:
        label_cortex.append(atlas_dir + '/tpl-fs32k' + f'/{cortex}.{hemi}.label.gii')

    # get parcel for both atlases
    atlas_cereb.get_parcel(label_cereb)
    atlas_cortex.get_parcel(label_cortex, unite_struct = unite_struct)

    # get connectivity stuff:
    if predict:
        conn_dir = os.path.join(cdata.conn_dir, conn_dataset, "train")
        weights = np.load(os.path.join(conn_dir, f"{cortex}_{conn_ses_id}_{conn_method}_logalpha_{log_alpha}_best_weights.npy"))
        scale = np.load(os.path.join(conn_dir, f'{conn_dataset}_scale.npy'))


    # loop through subjects and create a dataframe
    summary_list = []
    for sub in range(len(T.participant_id)):
        # get data for the current subject
        this_data_cereb = cdat[sub, :, :]
        this_data_cortex = ccdat[sub, :, :]

        # pass on the data with the atlas object to the aggregating function
        # get mean across voxels within parcel
        Y_parcel = agg_parcels(this_data_cereb,atlas_cereb.label_vector,fcn=np.nanmean)
        X_tessel = agg_parcels(this_data_cortex,atlas_cortex.label_vector,fcn=np.nanmean)

        # replacing nans with 0
        X_tessel = np.nan_to_num(X_tessel)
        Y_parcel = np.nan_to_num(Y_parcel)

        # use connectivity model to make predictions
        if predict:
            Yhat = predict_cerebellum(weights, scale, X_tessel, atlas_cereb, info, fwhm = 0) # getting the connectivity weights and scaling factor
            # get mean across voxels within parcel
            Yhat_parcel = agg_parcels(Yhat,atlas_cereb.label_vector,fcn=np.nanmean)
            # average over halves
            X_parcel = Yhat_parcel.copy()
        else:
            X_parcel = X_tessel.copy()
        
        # average over two halves (NOTE: This will be removed)
        X = np.nanmean(np.concatenate([X_parcel[info.half == 1, :, np.newaxis], X_parcel[info.half == 2, :, np.newaxis]], axis = 2), axis = 2)
        Y = np.nanmean(np.concatenate([Y_parcel[info.half == 1, :, np.newaxis], Y_parcel[info.half == 2, :, np.newaxis]], axis = 2), axis = 2)

        # looping over labels and doing regression for each corresponding label
        for ilabel in range(Y.shape[1]):
            info_sub = info.copy()
            info_sub = info_sub.loc[info.half == 1]
            print(f"- subject {T.participant_id[sub]} label {ilabel+1}")
            x = X[:, ilabel]
            y = Y[:, ilabel]

            coef, res, R2 = regressXY(x, y, fit_intercept = False)

            info_sub["sn"]    = T.participant_id[sub]
            info_sub["X"]     = x # X is either the cortical data or the predicted cerebellar activation
            info_sub["Y"]     = y
            info_sub["res"]   = res
            info_sub["coef"]  = coef * np.ones([len(info_sub), 1])
            info_sub["R2"]    = R2 * np.ones([len(info_sub), 1])
            info_sub["cortex"] = cortex * len(info_sub)
            info_sub['#region'] = (ilabel+1) * np.ones([len(info_sub), 1])

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0)
    return summary_df

# make cerebellar roi
def make_roi_cerebellum(cifti_img, info, threshold, atlas_space = "SUIT3", localizer = "Verbal2Back"):
    """
    creates label nifti for roi cerebellum
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    localizer (str) - name of the localizer
    Returns:
    nifti_img (nb.nifti) - nifti of the label created
    """
    # get suit data
    data = cifti_img.get_fdata()
    # get the row index corresponding to the contrast
    info_con = info.loc[info.cond_name == localizer]
    # get the map for the contrast of interest
    con_map  = data[info_con.cond_num.values -1, :]

    # get threshold value (ignoring nans)
    percentile_value = np.nanpercentile(con_map, q=threshold)

    # apply threshold
    thresh_data = con_map > percentile_value
    # convert 0 to nan
    thresh_data[thresh_data != False] = np.nan

    # create an instance of the atlas (will be used to convert data to nifti)
    atlas, a_info = am.get_atlas(atlas_space,atlas_dir)
    nifti_img = atlas.data_to_nifti(1*thresh_data)
    return nifti_img

# make cortical roi
def make_roi_cortex(cifti_img, info, threshold, localizer = "Verbal2Back"):
    """
    creates label giftis for left and right hemisphere of the cortex
    Args:
    cifti_img (nb.Cifti2) - cifti image of the extracted data
    info (pd.DataFrame) - pandas dataframe representing info for the dataset
    threshold (int) - percentile value used in thresholding
    localizer (str) - name of the localizer
    Returns:
    gifti_img (list) - list of label giftis for left and right hemispheres
    """
    # get data for left and right hemisphere
    data_list = nt.surf_from_cifti(cifti_img)

    # threshold and create label
    gifti_img = []
    for i, name in zip([0, 1], ['CortexLeft', 'CortexRight']):
        # get data for the hemisphere
        data = data_list[i]

        # get the contrast map
        # get the row index corresponding to the contrast
        info_con = info.loc[info.cond_name == localizer]
        # get the map for the contrast of interest
        con_map  = data[info_con.cond_num.values -1, :]

        # get threshold value (ignoring nans)
        percentile_value = np.nanpercentile(con_map, q=threshold)

        # apply threshold
        thresh_data = con_map > percentile_value
        # convert 0 to nan
        thresh_data[thresh_data != False] = np.nan
        # create label gifti
        gifti_img.append(nt.make_label_gifti(1*thresh_data.T, anatomical_struct=name))
    return gifti_img

# plotting maps on cerebellar flatmap
def prep_plot_cerebellum(img, 
                    atlas = "SUIT3",
                    threshold = None, 
                    alpha = 1,
                    scale = None, 
                    cmap = None, 
                    colorbar = True, 
                    render = "plotly"):
    """
    function to plot cerebellar map on the flat surface
    Args: 
        img (nb.cifti2 or str) - nb.cifti2 object or path to the cifti image to be plotted
                      pass one np.ndarray or cifti image as [img] to plot it without overlays
        atlas (str) - name of the space data were calculated in  (SUIT3 or SUIT2)
        threshold (list) - Hard thresholds to be applied to the images
        alpha (float) - floating point number for transparency (between 0 and 1)
        render (str) - renderer to be used for plotting. For papers use matplotlib (options: matplotlib, plotly)
    """
    # check if data is list and raise an error if it is not passed as a list
    if isinstance(img, str):
        img = nb.load(img)

    # get numpy array
    img_data = img.get_fdata()

    # project data to flatmap
    flat_map = suit.flatmap.vol_to_surf(img_data,space=atlas)
    fig = suit.flatmap.plot(flat_map, render=render, new_figure=True, cscale=scale, cmap=cmap, colorbar=colorbar)
    return fig


def plot_cortex_map(sub_list = ["group"],
                    dataset_name = "WMFS", 
                    ses_id = "ses-02", 
                    atlas = "fs32k", 
                    type = "condAll", 
                    surf = "veryinflated", 
                    threshold = None, 
                    cscale = None, 
                    cmap = ['coolwarm', 'viridis', 'binary'], 
                    alpha = [1, 0.6, 1], 
                    localizer = 'Verbal2Back',
                    outpath = None, 
                    cortex = "Icosahedron-1002_Sym.32k",
                    contrast = None, 
                    method = "L2Regression", 
                    logalpha = 8, 
                    interact = True, 
                    embed = False
                    ): 
    # get dataset class object
    Data = get_dataset_class(base_dir, dataset=dataset_name)
    # get info
    info = Data.get_info(ses_id,type)

    # prepare surfaces for plotting
    surfs = [os.path.join(atlas_dir, 'tpl-fs32k', f'tpl_fs32k_hemi-{hemi}_{surf}.surf.gii') for hemi in ['L', 'R']]

    # load the cifti
    for sub in sub_list:

        
        # plot:
        # all_views = ['lateral']
        # p = plot_surf(surf_lh = surfs[0], surf_rh = surfs[1], size = (1000, 1000), zoom=1.6, layout='grid', views = all_views)
        
        if contrast is not None:
            # cleaning: there's an extra space character appended to the names
            info['cond_name'] = info['cond_name'].map(lambda x: x.rstrip(' '))
            # get the index corresponding to the contrast
            info_con = info.loc[info.cond_name == contrast]
            idx = info_con["reg_id"].values - 1
            path_to_cifti = Data.data_dir.format(sub) + f'/{sub}_space-{atlas}_{ses_id}_{type}.dscalar.nii'
            # load the image
            img = nb.load(path_to_cifti)
            img_surf = nt.surf_from_cifti(img,
                                            struct_names=['CIFTI_STRUCTURE_CORTEX_LEFT',
                                                        'CIFTI_STRUCTURE_CORTEX_RIGHT'])
            # get data for current condition
            dat = [img_surf[h][idx, :] for h, hemi in enumerate(['L', 'R'])]
            # threshold the data?
            if threshold is not None:
                for h, hemi in enumerate(['L', 'R']):
                    dat[h] = dat[h][dat[h]<=threshold[0]] = np.nan
                    dat[h] = dat[h][dat[h]>=threshold[1]] = np.nan
            # p.add_layer({'left':dat[0], 'right': dat[1]},  cbar_label=contrast, color_range=[-1, 1], as_outline=False, cbar=True, cmap = "coolwarm", zero_transparent=True, alpha = 1) # cmap='YlOrBr_r',
        
        if localizer is not None:
            # prepare overlay map (ROI label file)
            roi_list = []
            for i, h in enumerate(['L', 'R']):
                label_file = nb.load(Data.atlas_dir + '/tpl-fs32k' + f'/{localizer}.32k.{h}.label.gii')
                roi_list.append(label_file.agg_data())
            # p.add_layer({'left': roi_list[0], 'right': roi_list[1]}, cmap='binary', as_outline=True, cbar=False, alpha = 1)
        
        if cortex is not None:
            # read in the connectivity
            path_to_cifti = cdata.conn_dir + "MDTB/train" + f'/{cortex}_{localizer}_{method}_{logalpha}.dscalar.nii'
            # load the image
            img_conn = nb.load(path_to_cifti)
            img_surf_conn = nt.surf_from_cifti(img_conn,
                                               struct_names=['CIFTI_STRUCTURE_CORTEX_LEFT',
                                                             'CIFTI_STRUCTURE_CORTEX_RIGHT'])

            # get data for current condition
            # dat_conn = [img_surf_conn[h] for h, hemi in enumerate(['L', 'R'])]

            # p.add_layer({'left':dat_conn[0].T, 'right': dat_conn[1].T},  cbar_label="weights", color_range=None, as_outline=False, cbar=True, cmap = "hot", zero_transparent=True, alpha = 0.5) # cmap='YlOrBr_r',
        

        # if interact:
        #     p.render()
        #     p.show(interactive = True, embed_nb = embed)

        # else:
        #     kws = {'location': 'bottom', 'label_direction': 360, 'decimals': 6,
        #             'fontsize': 9, 'n_ticks': 3, 'shrink': 0.15, 'aspect': 8,
        #             'draw_border': True}
        #     fig = p.build(cbar_kws=kws)
            
        #     plt.title("")
        #     plt.show()
    return dat, roi_list, img_surf_conn


   
# def rsa_ana(exp_name = 'fs', glm = 'glm4', 
#                 atlas = 'mdtb_10', 
#                 method = 'ridge', 
#                 cortex = 'tessels1002',
#                 mode = None,
#                 weighting = True,
#                 averaging = 'sess',
#                 fwhm = 0,
#                 region = 2,
#                 sn = const.return_subjs):
#     """
#     takes in glm, experiment name, and roi atlas and 
#     returns the beta estimates within each roi
#     """
#     # create directory object
#     exp_dirs = const.Dirs(exp_name = exp_name, glm = glm)

#     # get the best weights based on cortex and method
#     model_df = pd.read_csv(exp_dirs.weights_dir / f"best_models_{method}.csv")
#     model_name = model_df.loc[model_df["cortex_names"] == cortex]["models"].values[0]
#     eval_path = exp_dirs.base_dir / exp_name
#     pcm_path  = exp_dirs.conn_dir/ "pcm" / f"cerebellum_{model_name}"
#     if not os.path.exists(pcm_path):
#         os.makedirs(pcm_path)

#     # load the best weights
#     best_model = dd.io.load(exp_dirs.weights_dir / f"{model_name}.h5")

#     # get the smoothing matrix 
#     if fwhm != 0:
#         smooth_mat = get_smooth_mat(exp_name=exp_name, fwhm=fwhm)

#     # get the parcel numbers in suit space
#     atlas_file = os.path.join(exp_dirs.atlas, f"{atlas}.nii")
#     regions_suit = cdata.read_suit_nii(atlas_file)
#     regions = np.unique(regions_suit)


#     data =[] # to store dataset objects for rsatoolbox
#     data_pred = []
#     i = 0 # subject index
#     for s in sn:
        
#         # load data
#         X, X_info, Y, Y_info = get_data(exp_name = exp_name, 
#                                         glm=glm, 
#                                         weighting = weighting, 
#                                         averaging = averaging, 
#                                         cerebellum = 'cerebellum_suit_mdtb', 
#                                         cortex = cortex, 
#                                         subject = s)


#         # get the predictions using weights
#         Y_pred = np.dot(X, best_model['weights'].T)
#         if mode == "crossed":
#             Y_pred = np.r_[Y_pred[Y_info.sess == 2, :], Y_pred[Y_info.sess == 1, :]]

#         idx = regions_suit == region
#         Y_regs = (Y[:, idx])

#         Y_pred_regs = (Y_pred[:, idx])

#         nCond = Y_regs.shape[0]
#         nVox = Y_regs.shape[1]
#         des = {'subj':i+1}
#         obs_des = {'conds':Y_info.CN.values, 'session': Y_info.sess.values}
        
#         i=i+1

#         data.append(rsd.Dataset(measurements=Y_regs, 
#                                 descriptors = des, 
#                                 obs_descriptors = obs_des,
#                                 # channel_descriptors=chn_des
#                                 )
#                     )

#         data_pred.append(rsd.Dataset(measurements=Y_pred_regs, 
#                                 descriptors = des, 
#                                 obs_descriptors = obs_des,
#                                 # channel_descriptors=chn_des
#                                 )
#                     )

#         # noise_prec_diag = rst.data.noise.prec_from_measurements(data[0], obs_desc='conds', method='diag')

#     rdm_obs = rst.rdm.calc_rdm(data, method='crossnobis', descriptor='conds', cv_descriptor='session')
#     rdm_pred = rst.rdm.calc_rdm(data_pred, method='crossnobis', descriptor='conds', cv_descriptor='session')

#     rdm_obs_vec = rdm_obs.get_vectors()
#     rdm_pred_vec = rdm_pred.get_vectors()

#     D_dict = {}
#     D_dict['subj'] = []
#     D_dict['n'] = []
#     D_dict['obs'] = []
#     D_dict['pred'] = []
#     for s in np.arange(rdm_obs_vec.shape[0]):
#         for n in np.arange(rdm_obs_vec.shape[1]):
#             D_dict['subj'].append(s)
#             D_dict['n'].append(n)
#             D_dict['obs'].append(rdm_obs_vec[s, n])
#             D_dict['pred'].append(rdm_pred_vec[s, n])

#     # show individual RDMs
#     # fig, ax, ret_val = rst.vis.show_rdm(rdm_obs, show_colorbar='panel', rdm_descriptor='subj', pattern_descriptor = 'conds')
#     # fig, ax, ret_val = rst.vis.show_rdm(rdm_pred, show_colorbar='panel', rdm_descriptor='subj', pattern_descriptor = 'conds')

#     # rank transform RDMs
#     R_obs_subs = rst.rdm.rank_transform(rdm_obs)
#     R_pred_subs = rst.rdm.rank_transform(rdm_pred)

#     # show individual RDMs
#     # fig, ax, ret_val = rst.vis.show_rdm(R_obs_subs, show_colorbar='panel', rdm_descriptor='subj', pattern_descriptor = 'conds')
#     # fig, ax, ret_val = rst.vis.show_rdm(R_pred_subs, show_colorbar='panel', rdm_descriptor='subj', pattern_descriptor = 'conds')

#     # calculate cosine angle between RDMs (predicted and observed)
#     cc = rst.rdm.compare_cosine_cov_weighted(rdm_obs, rdm_pred)
#     fig, axes = plt.subplots(1, 1)
#     sns.heatmap(cc, annot=True, ax = axes)

#     # calculate group RDMs
#     rdm_obs_subs = rdm_obs.get_vectors()
#     rdm_pred_subs = rdm_pred.get_vectors()

#     rdm_obs_group = np.mean(rdm_obs_subs, axis = 0)
#     rdm_pred_group = np.mean(rdm_pred_subs, axis = 0)

#     # create group RDM objects
#     RDM_man_pred = rst.rdm.RDMs(rdm_pred_group, pattern_descriptors = rdm_obs.pattern_descriptors)
#     RDM_man_obs = rst.rdm.RDMs(rdm_obs_group, pattern_descriptors = rdm_pred.pattern_descriptors)
#     # show RDMs
#     # fig, ax, ret_val = rst.vis.show_rdm(RDM_man_obs, show_colorbar='panel', pattern_descriptor = 'conds')
#     # fig, ax, ret_val = rst.vis.show_rdm(RDM_man_pred, show_colorbar='panel', pattern_descriptor = 'conds')

#     R_obs_group = rst.rdm.rank_transform(RDM_man_obs)
#     R_pred_group = rst.rdm.rank_transform(RDM_man_pred)

#     fig, ax, ret_val = rst.vis.show_rdm(R_obs_group, show_colorbar='panel', rdm_descriptor='subj', pattern_descriptor = 'conds')
#     fig, ax, ret_val = rst.vis.show_rdm(R_pred_group, show_colorbar='panel', rdm_descriptor='subj', pattern_descriptor = 'conds')

#     # show MDS plots
#     # rst.vis.show_MDS(
#     #                 rdm_obs,
#     #                 pattern_descriptor='conds'

#     #                 )

#     # rst.vis.show_MDS(
#     #                 rdm_pred,
#     #                 pattern_descriptor='conds'

#     #                 )

#     rst.vis.show_MDS(
#                     RDM_man_obs,
#                     pattern_descriptor='conds'

#                     )

#     rst.vis.show_MDS(
#                     RDM_man_pred,
#                     pattern_descriptor='conds'

#                     )

#     return rdm_obs, rdm_pred, RDM_man_obs, RDM_man_pred,D_dict
