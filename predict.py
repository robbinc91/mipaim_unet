import nibabel

from utils.losses import *
from keras.models import load_model
from utils.basic_preprocess import *

import argparse

desc = 'Perform cerebellum segmentation using U-Net'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-o', '--output', required=True,
                    help='Output image')
parser.add_argument('-i', '--input', required=True,
                    help='Input image')
parser.add_argument('-m', '--model', required=False, default=None,
                    help='Model to use')
args = parser.parse_args()


def run_predict(input_path, output_path, _model):
    model_load = '/opt/models/cerebellum_full.h5'
    if _model is not None and _model != 'default':
        model_load = _model
    model = load_model(model_load, custom_objects={'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss})

    _input = histeq(to_uint8(get_data(input_path)))[None, None, ...]
    _output = model.predict(_input)
    _output = _output.squeeze()

    _img = nibabel.load(input_path)
    output_img = nibabel.Nifti1Image(_output, affine=_img.affine, header=_img.header)
    nibabel.save(output_img, output_path)


run_predict(args.input, args.output, args.model)
