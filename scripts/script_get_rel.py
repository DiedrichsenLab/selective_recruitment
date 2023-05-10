import numpy as np
import pandas as pd

import selective_recruitment.globals as gl
import Functional_Fusion.dataset as ds


def get_reliability_summary(dataset = "WMFS", ses_id = "ses-02", subtract_mean = True):
    """
    Calculates cross-validated reliability with runs as cross validating folds
    Args:
        dataset (str) - name of the dataset
        ses_id (str) - id assigned to the session
        subtract_mean(bool) - subtract mean before calculating the reliability?
    Returns:
        df (pd.DataFrame) - summary dataframe containing cross validated reliability measure
    """
    
    # get the datasets
    Data = ds.get_dataset_class(gl.base_dir, dataset=dataset)

    # loop over cortex and cerebellum
    D = []
    for atlas in ["SUIT3", "fs32k"]:

        # get the data tensor
        tensor, info, _ = ds.get_dataset(gl.base_dir,dataset,atlas = atlas, sess=ses_id,type='CondRun', info_only=False)

        # loop over conditions and calculate reliability
        for c, _ in enumerate(info.cond_name.loc[info.run == 1]):
            # get condition info data
            part_vec = info.run
            cond_vec = (info.cond_num == c+1)*(c+1)

            # get all the info from info to append to the summary dataframe
            info_cond = info.loc[(info.cond_num == c+1) & (info.run == 1)]

            r = ds.reliability_within_subj(tensor, part_vec, cond_vec,
                                            voxel_wise=False,
                                            subtract_mean=subtract_mean)

            # prep the summary dataframe
            R = pd.DataFrame()
            R["sn"] = Data.get_participants().participant_id
            R["R"] = np.mean(r, axis = 1)
            R["atlas"] = [atlas]*(r.shape[0])
            R.reset_index(drop = True, inplace=True)

            # get the rest of the info
            R_info = pd.concat([info_cond]*r.shape[0], axis = 0)
            # drop sn column from R_info
            R_info = R_info.drop(labels="sn", axis = 1)
            R_info.reset_index(drop = True, inplace=True)
            R = pd.concat([R, R_info], axis = 1, join = 'outer', ignore_index = False)

            D.append(R)
    
    df = pd.concat(D, axis = 0)
    
    return df
