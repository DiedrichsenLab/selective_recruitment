from math import comb
import numpy as np
import nibabel as nb 
import nitools as nt


def get_cortical_roi(model='x', startslice=12, endslice=16, threshold = 80,
                        hemisphere="L", roi_name='language', save_dir ='x'):

    # load cortical model
    image = nb.load(model)
    data_list = nt.surf_from_cifti(image)

    if hemisphere == 'L':
        data = data_list[0]
    else:
        data = data_list[1]

    data = data[startslice:endslice, :]

    # get threshold value (ignoring nans)
    percentile_value = np.nanpercentile(data, q=threshold)

    # apply threshold
    thresh_data = data > percentile_value

    # convert 0 to nan
    thresh_data[thresh_data != False] = np.nan

    combined_mask = thresh_data[0]

    for i in range(1, len(thresh_data)):
        combined_mask = np.logical_or(combined_mask, thresh_data[i])

    combined_mask = combined_mask.reshape(-1, 1)

    if hemisphere == 'L':
        anatomical_struct = 'CortexLeft'
    else:
        anatomical_struct = 'CortexRight'

    # create label gifti
    gifti = nt.make_label_gifti(1 * combined_mask, anatomical_struct=anatomical_struct)

    if hemisphere == 'L':
        nb.save(gifti, save_dir + '/tpl-fs32k' + f'/{roi_name}.32k.L.label.gii')
    else:
        nb.save(gifti, save_dir + '/tpl-fs32k' + f'/{roi_name}.32k.R.label.gii')


if __name__ == "__main__":
    save_dir = '/cifs/diedrichsen/data/Cerebellum/Language/atlases'
    model_path = '/cifs/diedrichsen/data/Cerebellum/connectivity/MDTB/train/Icosahedron-1002_Sym.32k_NettekovenSym68c32_L2Regression_8.dscalar.nii'

    get_cortical_roi(model = model_path, startslice=22, endslice=26, threshold= 90,
                    hemisphere='R',roi_name='md(D1-D4)_NettekovenSym68c32', save_dir= save_dir)