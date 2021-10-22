from utils import *

# VARIABLES
ROOT = "./datasets/mrbrains_full_2018/"  # BASE PATH TO MRBrainS18
LABEL = "Cerebellum"  # LABEL TO TRAIN FOR
EPOCHS = 400  # NUMBER OF EPOCHS
HAMMERS_ROOT = './datasets/hammers_full_2017/'
IMG_ROOT = './datasets/'

MY_ROOT = './datasets/mine/'
CERSEGSYS_ROOT = './datasets/cersegsys/'
CERSEGSYS_2_ROOT = './datasets/cersegsys_2/'
CERSEGSYS_3_ROOT = './datasets/cersegsys_3/'

ONLY_3X3_FILTERS = True
MNI_SHAPE = (1, 192, 224, 192)
x_idx_range = slice(32, 160)
y_idx_range = slice(30, 142)
z_idx_range = slice(10, 90)
REDUCED_MNI_SHAPE = (1, 128, 112, 80)
REDUCED_MNI_SHAPE_MINE = (1, 128, 128, 96)
REDUCED_MNI_SHAPE_CERSEGSYS_PARCELLATION = (1, 128, 80, 80)

MRBRAINS_SHAPE = (1, 240, 240, 48)

train_prt = [i for i in range(1, 19)] + [31, 34, 35, 36, 39]
dev_prt = [i for i in range(19, 25)] + [32, 37, 40]
test_prt = [i for i in range(25, 31)] + [33, 38, 41, 42]

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

