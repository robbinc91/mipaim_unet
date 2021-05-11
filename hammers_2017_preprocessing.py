import nibabel

from utils import *
from common import *
import SimpleITK as sitk



def do_preprocess(root):
    for i in range(1, 31):
        _name = 'a{:02d}'.format(i)

        # Bias field Correction -----------------
        #corrected_image, bias_field, log_bias_field = n4_bias_field_correction(root + 'images/' + _name + '.nii.gz')
        #sitk.WriteImage(corrected_image, root + 'pre/' + _name + '-bfc.nii.gz')
        # sitk.WriteImage(bias_field, root + 'pre/' + _name + '-bf.nii.gz')
        # sitk.WriteImage(log_bias_field, root + 'pre/' + _name + '-lbf.nii.gz')

        # Image Registration -------------------
        #transform, resampled = exhautive_registration(IMG_ROOT + 'registration/MNI152_T1_1mm.nii.gz', root + 'pre/' + _name + '-bfc.nii.gz')
        #sitk.WriteTransform(transform, root + 'pre/' + _name + '-transform.tfm')
        #sitk.WriteImage(resampled, root + 'pre/' + _name + '-reg.nii.gz')

        # labels resampling ----------------------
        #resampled_labels = applyTransform(IMG_ROOT + 'registration/MNI152_T1_1mm.nii.gz',
        #                                  root + 'images/' + _name + '-seg.nii.gz',
        #                                  root + 'pre/' + _name + '-transform.tfm')
        #sitk.WriteImage(resampled_labels, root + 'pre/' + _name + '-reg-seg.nii.gz')

        # Image and labels resample ----------------------
        #resized_img = conform_image(root + 'pre/' + _name + '-reg.nii.gz', (192, 224, 192), (1., 1., 1.))
        #nib.save(resized_img, root + 'pre/' + _name + '-reg-res.nii.gz')

        #resized_labels = conform_image(root + 'pre/' + _name + '-reg-seg.nii.gz', (192, 224, 192), (1., 1., 1.))
        #nib.save(resized_labels, root + 'pre/' + _name + '-reg-seg-res.nii.gz')

        # Skull Stripping and histogram equalization -----------------------
        #no_skull = get_data_with_skull_scraping(root + 'pre/' + _name + '-reg-res.nii.gz')
        #hist_equ = histeq(to_uint8(no_skull))
        #affine = nibabel.load(root + 'pre/' + _name + '-reg-res.nii.gz').affine
        #no_skull_img = nibabel.Nifti1Image(no_skull, affine)
        #hist_eq_img = nibabel.Nifti1Image(hist_equ, affine)

        #nibabel.save(no_skull_img, root + 'pre/' + _name + '-reg-res-no-skull.nii.gz')
        #nibabel.save(hist_eq_img, root + 'pre/' + _name + '-reg-res-no-skull-hist-eq.nii.gz')

        #img = nibabel.load(root + 'pre/' + _name + '-reg-res-no-skull-hist-eq.nii.gz')
        #nibabel.save(img, root + 'final/' + _name + '-pre.nii.gz')
        #seg = nibabel.load(root + 'pre/' + _name + '-reg-seg-res.nii.gz')
        #nibabel.save(seg, root + 'final/' + _name + '-labels.nii.gz')

        # Mask with only cerebellum
        seg = nibabel.load(root + 'pre/' + _name + '-reg-seg-res.nii.gz')
        #cerebl = np.array(seg.get_data() == 17).astype(np.uint8) + np.array(seg.get_data() == 18).astype(np.uint8)
        #cerebl = nibabel.Nifti1Image(cerebl, seg.affine)
        #nibabel.save(cerebl, root + 'final/' + _name + '-cerebellum.nii.gz')

        # Mask with only brainstem
        bst = np.array(seg.get_data() == 19).astype(np.uint8)
        bst = nibabel.Nifti1Image(bst, seg.affine)
        nibabel.save(bst, root + 'final/' + _name + '-brainstem.nii.gz')






if __name__ == '__main__':
    do_preprocess(HAMMERS_ROOT)