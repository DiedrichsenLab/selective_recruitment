from turtle import shape
from unicodedata import name
import numpy as np
import nibabel as nb 
import nitools as nt



image= nb.load('/cifs/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel/Atlases/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed_cortex-corr.dscalar.nii')
data= image.get_fdata()

data_LangL=data[13:17,:]
print(data_LangL.shape)


thresh= np.percentile(data_LangL,80)
data_LangL[data_LangL>thresh]=1
data_LangL[data_LangL<thresh]=0


gifti=nt.make_label_gifti(data_LangL, anatomical_struct='CortexLeft')
print(type(gifti))
nb.save(gifti, '/cifs/diedrichsen/data/Cerebellum/Language/Lang.L.label.gii')



