import nitools as nt
import numpy as np


def get_smooth_matrix(atlas, fwhm = 3):
    """
    calculates the smoothing matrix to be applied to data
    Args:
        atlas - experiment name (fs or wm)
        fwhm (float)  - fwhm of smoothing kernel
    Rerurns:
        smooth_mat (np.ndarray) - smoothing matrix
    """    
    
    # calculate euclidean distance
    if hasattr(atlas, "vox"): # for cerebellum
        euc_dist = nt.euclidean_dist_sq(atlas.vox, atlas.vox)
        smooth_mat = np.exp(-1/2 * euc_dist/(fwhm**2))
        smooth_mat = smooth_mat /np.sum(smooth_mat, axis = 1);  
    elif hasattr(atlas, "vertex"): # for cortex
        # get smoothing matrix for each hemi
        smooth_mat = []
        for h in [0, 1]:
            euc_dist = nt.euclidean_dist_sq(atlas.vertex[h], atlas.vertex[h])
            s_mat_hemi = np.exp(-1/2 * euc_dist/(fwhm**2))
            s_mat_hemi = s_mat_hemi /np.sum(s_mat_hemi, axis = 1)
            smooth_mat.append(s_mat_hemi)
    return smooth_mat