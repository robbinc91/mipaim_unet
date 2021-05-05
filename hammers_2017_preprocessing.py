from utils import *
from common import *
import SimpleITK as sitk


def do_preprocess(root):
    for i in range(12, 31):
        _name = 'a{:02d}'.format(i)
        corrected_image, bias_field, log_bias_field = n4_bias_field_correction(root + 'images/' + _name + '.nii.gz')
        sitk.WriteImage(corrected_image, root + 'pre/' + _name + '-bfc.nii.gz')
        # sitk.WriteImage(bias_field, root + 'pre/' + _name + '-bf.nii.gz')
        # sitk.WriteImage(log_bias_field, root + 'pre/' + _name + '-lbf.nii.gz')


if __name__ == '__main__':
    do_preprocess(HAMMERS_ROOT)