from utils import *

# VARIABLES
ROOT = "./datasets/mrbrains_full_2018/"  # BASE PATH TO MRBrainS18
LABEL = "Cerebellum"  # LABEL TO TRAIN FOR
EPOCHS = 400  # NUMBER OF EPOCHS
HAMMERS_ROOT = './datasets/hammers_full_2017/'
IMG_ROOT = './datasets/'


ONLY_3X3_FILTERS = True
MNI_SHAPE = (1, 192, 224, 192)
x_idx_range = slice(32, 160)
y_idx_range = slice(30, 142)
z_idx_range = slice(10, 90)
REDUCED_MNI_SHAPE = (1, 128, 112, 80)


MRBRAINS_SHAPE = (1, 240, 240, 48)


labels = {
    "Cortical gray matter": 1,
    "Basal ganglia": 2,
    "White matter": 3,
    "White matter lesions": 4,
    "Cerebrospinal fluid in the extracerebral space": 5,
    "Ventricles": 6,
    "Cerebellum": 7,
    "Brain stem": 8
}

label = labels[LABEL]


OPTIMIZER = 'adam'
IMAGE_ORDERING = 'channels_first'

