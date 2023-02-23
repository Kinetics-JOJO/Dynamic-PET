
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import nibabel as nib
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nb
import SimpleITK as sitk




#Read input
itk_img = sitk.ReadImage('./001/001_Crop_k1.nii')


#Median filter para setting
sitk_median = sitk.MedianImageFilter()
sitk_median.SetRadius(1)
sitk_median = sitk_median.Execute(itk_img)
sitk.WriteImage(sitk_median, './Other_method/Median/001_k1_median.nii')
print('Median Image save...')

nii_img =nb.load('./Other_method/Median/001_k1_median.nii')#load test.nii seems also work
nii_data = nii_img.get_data()
affine = nii_img.affine.copy()
hdr = nii_img.header.copy()

Recon = np.array(nii_data, dtype=np.float32)
Recon = np.reshape(Recon,[96,96,80])
Recon[Recon<0]=0
print('pixels less than o have been cropped...')
new_nii = nb.Nifti1Image(Recon, affine, hdr)
nb.save(new_nii,'./Other_method/Median/Final_001_k1_median.nii')
print('Image save...')




