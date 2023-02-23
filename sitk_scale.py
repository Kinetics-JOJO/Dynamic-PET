
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import nibabel as nib
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nb
import SimpleITK as sitk


def MaxminNormalization(x,set_min,set_max):
    x_max=np.max(x)
    x_min=np.min(x)
    k=(set_max-set_min)/(x_max-x_min)
    norX=k*x
    print('x_max=%s'%x_max)
    print('x_min=%s'%x_min)
    print('k=%s'%k)
    #x= (set_max-set_min)*(x-x_min)/(x_max-x_min)+set_min
    return norX



Patient_number='001'

nii_img_k1 =nb.load('./Data/NSSMIC/'+Patient_number+'/crop_image/crop_k1.nii')#load test.nii seems also work
nii_data_k1 = nii_img_k1.get_data()
affine_k1 = nii_img_k1.affine.copy()
hdr_k1 = nii_img_k1.header.copy()

nii_img_suv =nb.load('./Data/NSSMIC/'+Patient_number+'/crop_image/crop_im5.nii')#load test.nii seems also work
nii_data_suv = nii_img_suv.get_data()
affine_suv = nii_img_suv.affine.copy()
hdr_suv = nii_img_suv.header.copy()


nii_data_k1 =MaxminNormalization(nii_data_k1,0,50)

nii_data_suv=MaxminNormalization(nii_data_suv,0,3)

Recon_k1 = np.array(nii_data_k1, dtype=np.float32)
Recon_k1 = np.reshape(Recon_k1,[96,96,80])



Recon_suv = np.array(nii_data_suv, dtype=np.float32)
Recon_suv = np.reshape(Recon_suv,[96,96,80])




new_nii_k1 = nb.Nifti1Image(Recon_k1, affine_k1, hdr_k1)
new_nii_suv = nb.Nifti1Image(Recon_suv, affine_suv, hdr_suv)
nb.save(new_nii_k1,'./Data/scale/im5_scale/'+Patient_number+'/'+Patient_number+'_crop_k1_scale.nii')
nb.save(new_nii_suv,'./Data/scale/im5_scale/'+Patient_number+'/'+Patient_number+'_crop_im5_scale3.nii')
print('K1_Scale  Image save...')
print('Suv_Scale  Image save...')



