import numpy as np
import nibabel as nb 
import nitools as nt


# load probabilistic cortical dscalar (correlation based)
image= nb.load('/cifs/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel/Atlases/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed_cortex-corr.dscalar.nii')
data= image.get_fdata()

# slice parcels for language left
data=data[13:16,:]
print(type(data))
print(data.shape)

#threhshold data
thresholded_data = np.zeros_like(data)
for i in range(data.shape[0]):
    parcel = data[i, :]
    threshold = np.nanpercentile(parcel, 80)  # Select top 20% of values, ignoring NaN values
    thresholded_data[i, parcel >= threshold] = 1
    thresholded_data[i, parcel < threshold] = 0
    print(threshold)
# print(thresholded_data[,:])

# convert into label gifti and and save
gifti=nt.make_label_gifti(thresholded_data.T, anatomical_struct='CortexLeft') # use transpose of data (function needs vertices,columns)
nb.save(gifti, '/cifs/diedrichsen/data/Cerebellum/Language/Lang.L.label.gii')



