import numpy as np
import nibabel as nib
cv2_installed = True

try:
    import cv2 as cv
except ModuleNotFoundError:
    print('#####\n#-cv2 not available.\n#-can not perform histogram equalization.\n#-this may have bad consequences\n#####')
    cv2_installed = False


def get_data(path):
    return nib.load(path).get_data()


def histeq(data):
    if not cv2_installed:
        return data
    for slice_index in range(data.shape[2]):
        data[:, :, slice_index] = cv.equalizeHist(data[:, :, slice_index])
    return data


def to_uint8(data):
    data = data.astype(np.float)
    data[data < 0] = 0
    return ((data - data.min()) * 255.0 / data.max()).astype(np.uint8)
