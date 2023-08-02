from selective_recruitment.plotting import plot_parcels
from selective_recruitment.plotting import plot_connectivity_weight
from selective_recruitment.plotting import plot_mapwise_recruitment
from selective_recruitment.plotting import make_scatterplot

from selective_recruitment.scripts.script_mapwise import calc_ttest_mean

import selective_recruitment.regress as ra
import selective_recruitment.globals as gl
import selective_recruitment.data as ss
import selective_recruitment.region as sroi
import selective_recruitment.utils as sutil

import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds


from statsmodels.stats.anova import AnovaRM  # perform F test
import statsmodels
from scipy import stats


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SUITPy.flatmap as flatmap
import nitools as nt
import nilearn.plotting as plotting
from nilearn import datasets # this will be used to plot sulci on top of the surface
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from selective_recruitment.plotting import plot_parcels
from selective_recruitment.plotting import plot_connectivity_weight
from selective_recruitment.plotting import plot_mapwise_recruitment
from selective_recruitment.plotting import make_scatterplot

from selective_recruitment.scripts.script_mapwise import calc_ttest_mean

import selective_recruitment.regress as ra
import selective_recruitment.globals as gl
import selective_recruitment.data as ss
import selective_recruitment.region as sroi
import selective_recruitment.utils as sutil

import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds


from statsmodels.stats.anova import AnovaRM  # perform F test
import statsmodels
from scipy import stats


from pathlib import Path
import nibabel as nb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SUITPy.flatmap as flatmap
import nitools as nt
import nilearn.plotting as plotting
from nilearn import datasets # this will be used to plot sulci on top of the surface
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

from scipy.stats import norm, ttest_1samp



def group_test_cortex():
    """
    """
    
    wkdir = 'A:\data\Cerebellum\CerebellumWorkingMemory\selective_recruit'
    
    # create an instance of the dataset class
    Data = ds.get_dataset_class(base_dir = gl.base_dir, dataset = "WMFS")
    
    # get atlas objects
    atlas_cereb, atlas_info = am.get_atlas(atlas_str = "SUIT3", atlas_dir = gl.atlas_dir)
    atlas_cortex, atlas_info = am.get_atlas(atlas_str = "fs32k", atlas_dir = gl.atlas_dir)
    # get info 
    info = Data.get_info(ses_id="ses-02", type="CondAll", subj="group", fields=None)

    # define contrasts
    idx_enc = info.phase == 0
    c_enc = np.zeros(len(info.index))
    c_enc[idx_enc] = 1
    idx_ret = info.phase == 1
    c_ret = np.zeros(len(info.index))
    c_ret[idx_ret] = 1
    
    # get data for the cortex
    data_cortex,_,_ = ds.get_dataset(gl.base_dir,
                                    dataset = "WMFS",
                                    atlas="fs32k",
                                    sess="ses-02",
                                    subj=None,
                                    type = "CondAll",  
                                    smooth = 3)

    # get the contrast
    data_cortex_enc = c_enc @ data_cortex
    data_cortex_ret = c_ret @ data_cortex
    
    #
    

    _, pval_enc = ttest_1samp(data_cortex_enc, 0)
    z_val_enc = norm.isf(pval_enc)
    
    _, pval_ret = ttest_1samp(data_cortex_ret, 0)
    z_val_ret = norm.isf(pval_ret)
    
    
    enc_cii = atlas_cortex.data_to_cifti(z_val_enc.reshape(-1, 1).T)
    enc_img = nt.surf_from_cifti(enc_cii)
    
    ret_cii = atlas_cortex.data_to_cifti(z_val_ret.reshape(-1, 1).T)
    ret_img = nt.surf_from_cifti(ret_cii)
    
    
    surfs = [gl.atlas_dir + f'/tpl-fs32k/tpl_fs32k_hemi-{h}_inflated.surf.gii' for i, h in enumerate(['L', 'R'])]

    enc_fs_fig = []
    ret_fs_fig = []
    for h,hemi in enumerate(['left', 'right']):
        enc_fs_fig.append(plotting.plot_surf_stat_map(
                            surfs[h], enc_img[h], hemi=hemi,
                            colorbar=True, 
                            view = "lateral",
                            cmap="coolwarm",
                            engine='plotly',
                            symmetric_cbar = True,
                        ).figure)
        
        ret_fs_fig.append(plotting.plot_surf_stat_map(
                            surfs[h], ret_img[h], hemi=hemi,
                            colorbar=True, 
                            view = "lateral",
                            cmap="coolwarm",
                            engine='plotly',
                            symmetric_cbar = True,
                        ).figure)
    
    


    pass
    
    return




def calc_contrast():
    
    wkdir = 'A:\data\Cerebellum\CerebellumWorkingMemory\selective_recruit'
    
    # create an instance of the dataset class
    Data = ds.get_dataset_class(base_dir = gl.base_dir, dataset = "WMFS")
    
    # get atlas objects
    atlas_cereb, atlas_info = am.get_atlas(atlas_str = "SUIT3", atlas_dir = gl.atlas_dir)
    atlas_cortex, atlas_info = am.get_atlas(atlas_str = "fs32k", atlas_dir = gl.atlas_dir)
    # get info 
    info = Data.get_info(ses_id="ses-02", type="CondAll", subj="group", fields=None)

    # define contrasts
    idx_enc = info.phase == 0
    c_enc = np.zeros(len(info.index))
    c_enc[idx_enc] = 1
    idx_ret = info.phase == 1
    c_ret = np.zeros(len(info.index))
    c_ret[idx_ret] = 1

    # get data for the cerebellum
    data_cereb,_,_ = ds.get_dataset(gl.base_dir,
                                    dataset = "WMFS",
                                    atlas="SUIT3",
                                    sess="ses-02",
                                    subj=None,
                                    type = "CondAll",  
                                    smooth = 3)

    # get data for the cortex
    # data_cortex,_,_ = ds.get_dataset(gl.base_dir,
    #                                 dataset = dataset,
    #                                 atlas=atlas_fs,
    #                                 sess=ses_id,
    #                                 subj=subj,
    #                                 type = type,  
    #                                 smooth = smooth)

    # get the contrast
    data_cereb_enc = c_enc @ data_cereb
    data_cereb_ret = c_ret @ data_cereb

    # data_cortex_enc = c_enc @ data_cortex
    # data_cortex_ret = c_ret @ data_cortex
    
    # loop over subjects, get the data for the contrast and save it
    for i in range(data_cereb_enc.shape[0]):
        # get the data for the current subject for encoding
        data_cereb_enc_subj = data_cereb_enc[i,:]
        
        # get the data for the current subject for retrieval
        data_cereb_ret_subj = data_cereb_ret[i,:]
        
        # convert data for the encoding phase to nifti
        enc_nii = atlas_cereb.data_to_nifti(data_cereb_enc_subj)
        
        # conbert data for the retrieval phase to nifti
        ret_nii = atlas_cereb.data_to_nifti(data_cereb_ret_subj)
        
        # get the name for the subject 
        subj_name = f"sub-{i+1:02d}"
        
        # get the name for the encoding phase
        enc_name = f"{subj_name}_enc.nii"
        # get the name for the retrieval phase
        ret_name = f"{subj_name}_ret.nii"
        
        #save the nifti file for the encoding phase to the working directory
        nb.save(enc_nii, Path(wkdir, enc_name))

        #save the nifti file for the encoding phase to the working directory
        nb.save(ret_nii, Path(wkdir, ret_name))
    return


def surf_stat(df, 
              cortex_roi = "glasser_md", 
              phase = 0, 
              values = "X",
              mult_comp = "bonferroni",
              positive = True, # only takes the positive ts into account
              alpha = 0.01, corrected_map = False):
    """_summary_

    Args:
        df (_type_): _description_
        phase (int, optional): _description_. Defaults to 0.
        values (str, optional): _description_. Defaults to "X".
        mult_comp (str, optional): _description_. Defaults to "bonferroni".
        alpha (float, optional): _description_. Defaults to 0.01.
        corrected_map (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # get the data for the selected phase
    df = df.loc[df["phase"] == phase]
    
    # get the t-values for the selected phase
    D = pd.pivot_table(df, values = values, index = "sn", columns = "roi_name")
    
    # do a one-sample t-test for each region
    T_results= stats.ttest_1samp(D, np.zeros(len(D.columns)))

    # correct for multiple comparisons
    T_results_corr = statsmodels.stats.multitest.multipletests(T_results.pvalue, alpha=alpha, method=mult_comp, is_sorted=False, returnsorted=False)
    
    t_vals = T_results.statistic.copy()
    # get the regions with positive values
    if positive:
        pos_mask = t_vals>0
    else:
        pos_mask = np.bool_(np.ones(t_vals.shape))
    
    
    # get the significant regions
    sig_regs = D.columns[T_results_corr[0]]
    print(sig_regs)
    
    if corrected_map:
        # create an array with the t-values
        stat_array = np.zeros(t_vals.shape)
        stat_array[T_results_corr[0] & pos_mask] = t_vals[T_results_corr[0] & pos_mask]
    else:
        stat_array = t_vals.copy()
    # make atlas object first
    atlas_fs, _ = am.get_atlas("fs32k", gl.atlas_dir)

    # load the label file for the cortex
    label_fs = [gl.atlas_dir + f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]

    # get parcels for the neocortex
    labels, label_fs = atlas_fs.get_parcel(label_fs, unite_struct = False)

    surf_map = []
    for label in atlas_fs.label_list:
        # loop over regions within the hemisphere
        label_arr = np.zeros([label.shape[0], 1])
        
        # transfer the t-values to the surface
        # get the t-value for the current region
        for i, l in enumerate(np.unique(label)):
            if i == 0:
                tval = np.nan
            else:
                tval = stat_array[i-1]
            # get the vertices of the current region
            vert = np.where(label == l)[0]
                
            # assign the t-value to the vertices of the current region
            label_arr[vert] = tval

        surf_map.append(label_arr.T)
        
    cifti_img = atlas_fs.data_to_cifti(surf_map)
        

    return nt.surf_from_cifti(cifti_img), sig_regs


def flatmap_stat(df, 
             phase = 0, 
             values = "X",
             mult_comp = "bonferroni",
             alpha = 0.01):
    """_summary_

    Args:
        df (_type_): _description_
        phase (int, optional): _description_. Defaults to 0.
        values (str, optional): _description_. Defaults to "X".
        mult_comp (str, optional): _description_. Defaults to "bonferroni".
        alpha (float, optional): _description_. Defaults to 0.01.
    """
    # get the data for the selected phase
    df = df.loc[df["phase"] == phase]
    
    # get the t-values for the selected phase
    D = pd.pivot_table(df, values = values, index = "sn", columns = "roi_name")
    
    # do a one-sample t-test for each region
    T_results= stats.ttest_1samp(D, np.zeros(len(D.columns)))

    # correct for multiple comparisons
    T_results_corr = statsmodels.stats.multitest.multipletests(T_results.pvalue, alpha=alpha, method=mult_comp, is_sorted=False, returnsorted=False)
    
    # get the significant regions
    sig_regs = D.columns[T_results_corr[0]]
    print(sig_regs)
    stat_array = np.zeros(T_results.statistic.shape)
    stat_array[T_results_corr[0]] = T_results.statistic[T_results_corr[0]]
    ## make atlas object first
    atlas, ainfo = am.get_atlas("SUIT3", gl.atlas_dir)

    # load the label
    label_file = gl.atlas_dir + f"/tpl-SUIT/atl-NettekovenSym32_space-SUIT_dseg.nii"

    # get parcels for the neocortex
    labels, label = atlas.get_parcel(label_file)
    # loop over regions within the hemisphere
    label_arr = np.zeros([labels.shape[0]])
    
    # transfer the t-values to the surface
    # get the t-value for the current region
    for i in np.unique(label):
        if i == 0:
            tval = 0
        else:
            tval = stat_array[i-1]
        # get the vertices of the current region
        vert = np.where(label == i)[0]
            
        # assign the t-value to the vertices of the current region
        label_arr[vert] = tval

    print(label_arr)
    # convert to nifti
    vol_nii = atlas.data_to_nifti(label_arr.T) 
    
    # map to surface
    flat_arr = flatmap.vol_to_surf([vol_nii], space = 'SUIT', ignore_zeros=True, stats="mode")

    return flat_arr


def get_stat_md():
    """_summary_
    """
    
    # get the data for the regions starting with D
    cereb1_df = pd.read_csv(Path(wkdir) / "NettekovenSym32_df.csv", index_col = 0)
    cereb2_df = pd.read_csv(Path(wkdir) / "NettekovenSym32AP_df.csv", index_col = 0)
    return


if __name__ == "__main__":
    # setting working directory
    wkdir = 'A:\data\Cerebellum\CerebellumWorkingMemory\selective_recruit'
    cortex_df = pd.read_csv(Path(wkdir) / "glasser_df.csv", index_col = 0)
    # limiting the statistical analysis to the multiple demand network
    selected_regions = ["a9-46v", "p10p", "a10p", "11l", "a47r", 
                        "p47r", "FOP5", "AVl", "6r", "IFJp", "8C", "p9-46v", 
                        "i6-8", "s6-8", "AIP", "IP2", "IP1", 
                        "LIPd", "MIP", "PGs", "PFm", "TE1m", "TE1p", 
                        "POS2", "SCEF", "8BM", "a32pr", "d32"]

    # print(len(selected_regions))

    selected_names = [ f"{hemi}_{s}_ROI" for s in selected_regions for hemi in ["L", "R"]]

    df = cortex_df.loc[np.isin(cortex_df["roi_name"], selected_names)]
    surf_map_enc = surf_stat(df, 
                      phase = 0, 
                      values = "X",
                      positive = True,
                      mult_comp = "bonferroni",
                      alpha = 0.01)
    pass