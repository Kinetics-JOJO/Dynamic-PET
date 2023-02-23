
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import nibabel as nib
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nb
import SimpleITK as sitk

Patient_number = ['001','020','104','013','018','002','077']
tb=range(0,7)
for batch_i in tb:

    itk_img = sitk.ReadImage('./Data/NSSMIC/'+Patient_number[batch_i]+'/crop_image/crop_k1.nii')


    #Gassuian para setting
    sitk_gassuian = sitk.SmoothingRecursiveGaussianImageFilter()
    sitk_gassuian.SetSigma(1.8)
    sitk_gassuian.NormalizeAcrossScaleOff()  
    sitk_gassuian = sitk_gassuian.Execute(itk_img)
    sitk.WriteImage(sitk_gassuian, './Other_method/Scaled_result/Gassuian/im5/'+Patient_number[batch_i]+'/'+Patient_number[batch_i]+'_k1_Gassuian1.8.nii')

    print('Gassuian Image save...')

    nii_img =nb.load('./Other_method/Scaled_result/Gassuian/im5/'+Patient_number[batch_i]+'/'+Patient_number[batch_i]+'_k1_Gassuian1.8.nii')#load test.nii seems also work
    nii_data = nii_img.get_data()
    affine = nii_img.affine.copy()
    hdr = nii_img.header.copy()

    Recon = np.array(nii_data, dtype=np.float32)
    Recon = np.reshape(Recon,[96,96,80])
    Recon[Recon<0]=0
    print('pixels less than o have been cropped...')
    new_nii = nb.Nifti1Image(Recon, affine, hdr)
    nb.save(new_nii,'./Other_method/Orgin_k1base/Gassuian/'+Patient_number[batch_i]+'/Inverse_'+Patient_number[batch_i]+'_im5_k1_Gassuian1.8.nii')
    print('Image  %d save...' %(batch_i))


print('ALL Image  save...')


