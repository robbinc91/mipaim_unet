
import numpy as np
import nibabel as nib

import argparse

desc = 'Crop nifti file'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-o', '--output', required=True,
                    help='Output cropped image')
parser.add_argument('-i', '--input', required=True,
                    help='Input image')
args = parser.parse_args()

TEMPLATE_SHAPE = (193, 229, 193)
CROP_AREA = (slice(36, 164), slice(80, 208), slice(0, 96))



def crop_file(input_path, bbox, output_path):
    image = nib.load(input_path)
    roi = np.asarray(image.dataobj[bbox[0], bbox[1], bbox[2]])
    cropped_nifti = nib.Nifti1Image(roi, affine=image.affine, header=image.header)
    nib.save(cropped_nifti, output_path)


crop_file(args.input, CROP_AREA, args.output)



