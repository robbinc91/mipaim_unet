#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse

desc = 'Blur a mask using Gaussian filter to create weight image'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-m', '--mask', required=True,
                    help='The mask to calcualte weight image from')
parser.add_argument('-o', '--output', required=True,
                    help='The weight image')
parser.add_argument('-i', '--image', required=False, default=None,
                    help='if specified, use the header of the input image')
parser.add_argument('-s', '--sigma', type=int, default=3,
                    help='The sigma of the filter in mm')
parser.add_argument('-f', '--factor', type=float, default=0,
                    help='Structure element enlarging factor')
args = parser.parse_args()


import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.filters import gaussian_filter

    
obj = nib.load(args.mask)
mask = obj.get_data().astype(np.float32)

x, y, z = mask.shape
rx, ry, rz = obj.header.get_zooms()
interp = RegularGridInterpolator((range(x), range(y), range(z)), mask)
xi = np.arange(0, x-1, 1/rx)
yi = np.arange(0, y-1, 1/ry)
zi = np.arange(0, z-1, 1/rz)
xgrid, ygrid, zgrid = np.meshgrid(xi, yi, zi, indexing='ij')
points = np.hstack([xgrid.flatten()[..., None],
                    ygrid.flatten()[..., None],
                    zgrid.flatten()[..., None]])
interpolated = interp(points).reshape(xgrid.shape) > 0.5

radius = int((args.factor * args.sigma - 1) // 2  * 2 + 1)
size = 2 * radius + 1
coords = np.meshgrid(*[range(size)]*3, indexing='ij')
se = (coords[0]-radius)**2 + (coords[1]-radius)**2 \
        + (coords[2]-radius)**2 <= radius ** 2

if np.prod(se.shape) > 0:
    interpolated = binary_erosion(interpolated, structure=se)
blurred = gaussian_filter(interpolated.astype(np.float32), args.sigma,
                          mode='constant')

interp = RegularGridInterpolator((xi, yi, zi), blurred, bounds_error=False,
                                 fill_value=0)
xgrid, ygrid, zgrid = np.meshgrid(range(x), range(y), range(z), indexing='ij')
points = np.hstack([xgrid.flatten()[..., None],
                    ygrid.flatten()[..., None],
                    zgrid.flatten()[..., None]])
resampled = interp(points).reshape(xgrid.shape)

if args.image is not None:
    obj = nib.load(args.image)
output = nib.Nifti1Image(resampled, obj.affine, obj.header)
output.set_data_dtype(np.float32)
output.to_filename(args.output)
