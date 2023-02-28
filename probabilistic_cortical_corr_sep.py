from turtle import shape
from unicodedata import name
import numpy as np
import nibabel as nb 
import nitools as nt



image= nb.load('/cifs/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel/Atlases/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed_cortex-corr.dscalar.nii')
data= image.get_fdata()

data=data[13:16,:]
print(type(data))
print(data.shape)


thresholded_data = np.zeros_like(data)
for i in range(data.shape[0]):
    parcel = data[i, :]
    threshold = np.nanpercentile(parcel, 80)  # Select top 20% of values, ignoring NaN values
    thresholded_data[i, parcel >= threshold] = 1
    thresholded_data[i, parcel < threshold] = 0
    print(threshold)

print(thresholded_data[0])


gifti=nt.make_label_gifti(thresholded_data, anatomical_struct='CortexLeft')
nb.save(gifti, '/cifs/diedrichsen/data/Cerebellum/Language/Lang.L.label.gii')



