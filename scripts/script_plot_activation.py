from pathlib import Path
import numpy as np

import SUITPy.flatmap as flatmap
from nilearn import plotting
import nitools as nt

import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am

import selective_recruitment.globals as gl

import matplotlib.pyplot as plt
import nibabel as nb

def plot_activation_map(dataset = "WMFS", 
                         ses_id = "ses-02", 
                         subj = "group",
                         type = "CondAll", 
                         atlas_space = "SUIT3", 
                         contrast_name = "average", 
                         cmap = "coolwarm",
                         cscale = [-0.2, 0.2],  
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
        img_flat = flatmap.vol_to_surf([img_nii], stats='nanmean', space = 'SUIT', ignore_zeros=True)
        ax = flatmap.plot(data=img_flat, 
                          render="plotly", 
                          hover='auto', 
                          cmap = cmap, 
                          colorbar = True, 
                          bordersize = 1, 
                          cscale = cscale)

    elif atlas_space == "fs32k":
        # get inflated cortical surfaces
        surfs = [gl.atlas_dir + f'/tpl-fs32k/tpl_fs32k_hemi-{h}_inflated.surf.gii' for i, h in enumerate(['L', 'R'])]

        # first convert to cifti
        img_cii = atlas.data_to_cifti(dat_con.reshape(-1, 1).T)
        img_con = nt.surf_from_cifti(img_cii)
        
        ax = []
        for h in [0, 1]:
            fig = plotting.plot_surf_stat_map(
                                            surfs[0], img_con[0], hemi='left',
                                            # title='Surface left hemisphere',
                                            colorbar=True, 
                                            view = 'lateral',
                                            cmap="coolwarm",
                                            engine='plotly',
                                            symmetric_cbar = True,
                                            vmax = cscale[1]
                                        )

            ax.append(fig.figure)
    return ax


if __name__ == "__main__":
    plot_activation_map(dataset = "WMFS", 
                         ses_id = "ses-02", 
                         subj = "group",
                         type = "CondAll", 
                         atlas_space = "fs32k", 
                         contrast_name = "average", 
                         cmap = "coolwarm",
                         cscale = [-0.2, 0.2],  
                         smooth = None)
    pass