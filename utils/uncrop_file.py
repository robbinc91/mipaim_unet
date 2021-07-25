
import numpy as np
import nibabel as nib

import argparse

desc = 'Uncrop nifti file'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-o', '--output', required=True,
                    help='Output uncropped image')
parser.add_argument('-i', '--input', required=True,
                    help='Input image')
args = parser.parse_args()

TEMPLATE_SHAPE = (193, 229, 193)
CROP_AREA = (slice(36, 164), slice(80, 208), slice(0, 96))


def uncrop_file(input_path, source_shape, source_bbox, output_path):
    _image = nib.load(input_path)
    image = _image.get_data()
    uncropped = np.zeros(source_shape, dtype=image.dtype)
    uncropped[tuple(source_bbox)] = image[:,:,:]
    uncropped_nifti = nib.Nifti1Image(uncropped, affine=_image.affine, header=_image.header)
    nib.save(uncropped_nifti, output_path)


uncrop_file(args.input, TEMPLATE_SHAPE, CROP_AREA, args.output)