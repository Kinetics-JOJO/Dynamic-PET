"""
==============================================
Denoise images using Non-Local Means (NLMEANS)
==============================================

Using the non-local means filter [Coupe08]_ and [Coupe11]_ and  you can denoise
3D or 4D images and boost the SNR of your datasets. You can also decide between
modeling the noise as Gaussian or Rician (default).

"""
#coding:utf-8
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
# from dipy.data import get_fnames
# from dipy.io.image import load_nifti
from skimage import transform
from PIL import Image
import SimpleITK as sitk



# dwi_fname, dwi_bval_fname, dwi_bvec_fname = get_fnames('sherbrooke_3shell')
#data, affine = load_nifti(dwi_fname)

nii_img =nib.load('./001/im5/001_crop_k1_scale.nii')#load test.nii seems also work
nii_data = nii_img.get_data()
affine = nii_img.affine.copy()
hdr = nii_img.header.copy()


mask = nii_data 

# We select only one volume for the example to run quickly.
#nii_data = nii_data[..., 1]

print("vol size", nii_data.shape)

# lets create a noisy data with Gaussian data

"""
In order to call ``non_local_means`` first you need to estimate the standard
deviation of the noise. We use N=4 since the Sherbrooke dataset was acquired
on a 1.5T Siemens scanner with a 4 array head coil.
"""

sigma = estimate_sigma(nii_data, N=1)

t = time()

"""
Calling the main function ``non_local_means``
"""

t = time()

den = nlmeans(nii_data, sigma=sigma, mask=mask, patch_radius=0.5,
              block_radius=1, rician=False)

print("total time", time() - t)
"""
Let us plot the axial slice of the denoised output
"""

# axial_middle = data.shape[2] // 2

# before = data[:, :, axial_middle].T
# after = den[:, :, axial_middle].T

# difference = np.abs(after.astype(np.float64) - before.astype(np.float64))

# difference[~mask[:, :, axial_middle].T] = 0


# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(before, cmap='gray', origin='lower')
# ax[0].set_title('before')
# ax[1].imshow(after, cmap='gray', origin='lower')
# ax[1].set_title('after')
# ax[2].imshow(difference, cmap='gray', origin='lower')
# ax[2].set_title('difference')

# plt.savefig('denoised.png', bbox_inches='tight')


# """
# .. figure:: denoised.png
#    :align: center

#    **Showing axial slice before (left) and after (right) NLMEANS denoising**
# """
new_nii = nib.Nifti1Image(den, affine, hdr)
nib.save(new_nii, 'Other_method/NLM/im5/NLM3D_im5_001.nii')
print('NLM denoised Image save...')