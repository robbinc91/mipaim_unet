#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

desc = 'Calcualte N4 skrinking factor'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('input', help='The input image to perform N4')
args = parser.parse_args()

import nibabel as nib
import numpy as np

obj = nib.load(args.input)
num_pixels = np.prod(obj.shape) ** (1/3)
shrinking_factor = np.round(num_pixels / 200 * 4).astype(int)
print(shrinking_factor)
