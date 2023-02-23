
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nb
import SimpleITK as sitk
import os



def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``

    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)
    
    return ornt_transf, ornt_init, ornt_fin

def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
    Performs the reorientation (changing order of axes)

    :param data_array: 3D Array to reorient
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    return nib.orientations.apply_orientation(data_array, ornt_transf)

# I test the code by the following simple demo, and it works.

lits_path = './001/001_crop_k1_scale.nii'  # RAS 

kits_path = './Other_method/BM4D/BM4D_001_im5_K1.nii'  # LPI

lits_nii = nib.load(lits_path)
lits_data = lits_nii.get_data() 
lits_axcodes = tuple(nib.aff2axcodes(lits_nii.affine)) # ('R', 'A', 'S')
lits_hdr = lits_nii.header.copy()

kits_nii = nib.load(kits_path) 
kits_data = kits_nii.get_data() 
kits_axcodes = tuple(nib.aff2axcodes(kits_nii.affine)) # ('I', 'P', 'L')
kits_hdr = kits_nii.header.copy()

new_kits_img = do_reorientation(kits_data, kits_axcodes, lits_axcodes) 
#new_lits_img = do_reorientation(lits_data, lits_axcodes, kits_axcodes) 

new_nii = nb.Nifti1Image(new_kits_img, lits_nii.affine, lits_hdr)
nib.save(new_nii,'./Other_method/BM4D/BM4D_001_im5_K1_FLIPRPS.nii')
print("Horizontal_RPS filp done...")