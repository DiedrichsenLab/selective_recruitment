from math import comb
import numpy as np
import nibabel as nb 
import nitools as nt


image= nb.load ('/cifs/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel/Atlases/NettekovenSym68c32_dseg.nii')
data = image.get_fdata()


data = image.get_fdata()
data = ((data >= 28.9) & (data <= 32.1)).astype(int)


new_image=nb.Nifti1Image(data, affine=image.affine, header=image.header)
nb.save(new_image, f'/cifs/diedrichsen/data/Cerebellum/Language/atlases/tpl-SUIT/atl-language(S1R-S4R)_NettekovenSym68c32_space-SUIT_dseg.nii') 