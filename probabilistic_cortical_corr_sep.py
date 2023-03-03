import numpy as np
import nibabel as nb 
import nitools as nt


# load fs32k L and R masks 
fs32kL_image=nb.load('/cifs/diedrichsen/data/FunctionalFusion/Atlases/tpl-fs32k/tpl-fs32k_hemi-L_mask.label.gii')
fs32kL = nt.get_gifti_data_matrix(fs32kL_image)
fs32kR_image=nb.load('/cifs/diedrichsen/data/FunctionalFusion/Atlases/tpl-fs32k/tpl-fs32k_mask.R.label.gii')
fs32kR = nt.get_gifti_data_matrix(fs32kR_image)
print(f'mask shape is {fs32kL.shape}')


# # create left hemisphere mask
left_hemi_mask = (fs32kL == 1) | (fs32kL == 0)
right_hemi_mask = (fs32kR == 1) | (fs32kR ==0)

# load probabilistic cortical dscalar (correlation based)
image= nb.load('/cifs/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel/Atlases/NettekovenSym34_cortex_connmodel.dscalar.nii')
data= image.get_fdata()

# slice parcels for language left
data = data[12:16, :]
print(f'shape of data is{data.shape}')


left_hemi_data = np.zeros((4, 32492))
for i in range(data.shape[0]):
    indices = np.where(left_hemi_mask)[0]
    left_hemi_data[i] = data[i, indices]
print(f'seperated  data(3parcels) shape is {left_hemi_data.shape}')

# gifti=nt.make_label_gifti(left_hemi_data.T, anatomical_struct='CortexLeft') # use transpose of data (function needs vertices,columns)
# nb.save(gifti, '/cifs/diedrichsen/data/Cerebellum/Language/languageparcels.label.gii')

# threshold data
thresholded_data = np.zeros_like(left_hemi_data)
for i in range(left_hemi_data.shape[0]):
    parcel = left_hemi_data[i, :]
    threshold = np.nanpercentile(parcel, 80)  # Select top 20% of values, ignoring NaN values
    thresholded_data[i, parcel >= threshold] = 1
    thresholded_data[i, parcel < threshold] = 0
print(f'thresholded data(3parcels) and seperated shape is{thresholded_data.shape}')


parcel_1 = left_hemi_data[0].reshape(-1, 1)
parcel_2 = left_hemi_data[1].reshape(-1, 1)
parcel_3 = left_hemi_data[2].reshape(-1, 1)
parcel_4 = left_hemi_data[3].reshape(-1, 1)


print(parcel_1.shape)

# # convert into label gifti and and save
# gifti=nt.make_label_gifti(data.T, anatomical_struct='CortexLeft') # use transpose of data (function needs vertices,columns)
# nb.save(gifti, '/cifs/diedrichsen/data/Cerebellum/Language/Lang.L.label.gii')
# new_image=nb.Nifti1Image(data[0,:], affine=np.imag.affine, header=image.header)
# nb.save(new_image, '/cifs/diedrichsen/data/Cerebellum/Language/Lang.L.nii')




