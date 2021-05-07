from utils import *
from common import *
import SimpleITK as sitk


def do_preprocess(root):
    for i in range(1, 31):
        _name = 'a{:02d}'.format(i)
        # Bias field Correction
        #corrected_image, bias_field, log_bias_field = n4_bias_field_correction(root + 'images/' + _name + '.nii.gz')
        #sitk.WriteImage(corrected_image, root + 'pre/' + _name + '-bfc.nii.gz')
        # sitk.WriteImage(bias_field, root + 'pre/' + _name + '-bf.nii.gz')
        # sitk.WriteImage(log_bias_field, root + 'pre/' + _name + '-lbf.nii.gz')

        # Image Registration
        #transform, resampled = exhautive_registration(IMG_ROOT + 'registration/MNI152_T1_1mm.nii.gz', root + 'pre/' + _name + '-bfc.nii.gz')
        #sitk.WriteTransform(transform, root + 'pre/' + _name + '-transform.tfm')
        #sitk.WriteImage(resampled, root + 'pre/' + _name + '-reg.nii.gz')

        # labels resampling
        resampled_labels = applyTransform(IMG_ROOT + 'registration/MNI152_T1_1mm.nii.gz',
                                          root + 'images/' + _name + '-seg.nii.gz',
                                          root + 'pre/' + _name + '-transform.tfm')
        sitk.WriteImage(resampled_labels, root + 'pre/' + _name + '-reg-seg.nii.gz')



if __name__ == '__main__':
    do_preprocess(HAMMERS_ROOT)