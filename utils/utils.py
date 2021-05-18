import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
import keras
from utils.preprocess import to_uint8, get_data
import math


def visualize(PATH, View="Axial_View", cmap=None):
    """
    Visualize Image

    Parameters
    ----------
    :param cmap:
    :param PATH: Path to *nii.gz image or image as numpy array
    :param View: Sagittal_View or Axial_View or Coronal_View

    """

    plt.ion()
    view = View.lower()
    v = {
        "sagittal_view": 0,
        "axial_view": 2,
        "coronal_view": 1,
    }
    if view not in ["sagittal_view", "axial_view", "coronal_view"]:
        print("Enter Valid View")
        return
    if type(PATH) == np.str_ or type(PATH) == str:
        img = nib.load(PATH)
        img1 = img.get_fdata()
    elif type(PATH) == np.ndarray and len(PATH.shape) == 3:
        img1 = PATH
    else:
        print("ERROR: Input valid type 'PATH'")
        return
    for i in range(img1.shape[v[view]]):
        plt.cla()
        if v[view] == 0:
            plt.imshow(img1[i, :, :], cmap=cmap)
        elif v[view] == 1:
            plt.imshow(img1[:, i, :], cmap=cmap)
        else:
            plt.imshow(img1[:, :, i], cmap=cmap)
        plt.show()
        plt.pause(0.1)

    print(f"Shape of image:{img1.shape}")


def mrbrains2018_data_train(root='./', t1=True, flair=False, ir=False):
    T1path = [root + 'training/1/pre/reg_T1.nii.gz',
              root + 'training/4/pre/reg_T1.nii.gz',
              root + 'training/7/pre/reg_T1.nii.gz',
              root + 'training/14/pre/reg_T1.nii.gz',
              root + 'training/5/pre/reg_T1.nii.gz',
              root + 'training/070/pre/reg_T1.nii.gz']

    segpath = [root + 'training/1/segm.nii.gz',
               root + 'training/4/segm.nii.gz',
               root + 'training/7/segm.nii.gz',
               root + 'training/14/segm.nii.gz',
               root + 'training/5/segm.nii.gz',
               root + 'training/070/segm.nii.gz']

    return T1path, segpath

def hammers_2017_data_train(root):
    T1Path = [root + 'pre/' + 'a{:02d}-reg-res.nii.gz'.format(i) for i in range(1, 25)]
    segpath = [root + 'pre/' + 'a{:02d}-reg-seg-res.nii.gz'.format(i) for i in range(1, 25)]

    return T1Path, segpath

def hammers_2017_data_preprocessed_train(root):
    T1Path = [root + 'final/' + 'a{:02d}-pre.nii.gz'.format(i) for i in range(1, 25)]
    segpath = [root + 'final/' + 'a{:02d}-cerebellum.nii.gz'.format(i) for i in range(1, 25)]

    return T1Path, segpath

def hammers_2017_data_preprocessed_train_reduced(root):
    T1Path = [root + 'reduced/' + 'a{:02d}-pre.nii.gz'.format(i) for i in range(1, 25)]
    segpath = [root + 'reduced/' + 'a{:02d}-cerebellum.nii.gz'.format(i) for i in range(1, 25)]

    return T1Path, segpath

def mrbrains2018_data_val(root="./", t1=True, flair=False, ir=False):

    T1_val = root + 'training/148/pre/reg_T1.nii.gz'
    segm_val = root + 'training/148/segm.nii.gz'

    return T1_val, segm_val


def hammers_2017_data_val(root):
    T1_val = [root + 'pre/' + 'a{:02d}-reg-res.nii.gz'.format(i) for i in range(19, 25)]
    segm_val = [root + 'pre/' + 'a{:02d}-reg-seg-res.nii.gz'.format(i) for i in range(19, 25)]

    return T1_val, segm_val

def data_train(root="./"):
    """
    Return:
    T1path - list of paths to registered Bias Normalized T1
    FLAIRpath - list of paths to registered Bias Normalized FLAIR
    IRpath - list of paths to registered Bias Normalized IR
    segpath - list of paths to segmentations
    """

    T1path = [root + 'training/1/pre/reg_T1.nii.gz',
              root + 'training/4/pre/reg_T1.nii.gz',
              root + 'training/7/pre/reg_T1.nii.gz',
              root + 'training/14/pre/reg_T1.nii.gz',
              root + 'training/5/pre/reg_T1.nii.gz',
              root + 'training/070/pre/reg_T1.nii.gz']
    FLAIRpath = [root + 'training/1/pre/FLAIR.nii.gz',
                 root + 'training/4/pre/FLAIR.nii.gz',
                 root + 'training/7/pre/FLAIR.nii.gz',
                 root + 'training/14/pre/FLAIR.nii.gz',
                 root + 'training/5/pre/FLAIR.nii.gz',
                 root + 'training/070/pre/FLAIR.nii.gz']
    IRpath = [root + 'training/1/pre/reg_IR.nii.gz',
              root + 'training/4/pre/reg_IR.nii.gz',
              root + 'training/7/pre/reg_IR.nii.gz',
              root + 'training/14/pre/reg_IR.nii.gz',
              root + 'training/5/pre/reg_IR.nii.gz',
              root + 'training/070/pre/reg_IR.nii.gz']
    segpath = [root + 'training/1/segm.nii.gz',
               root + 'training/4/segm.nii.gz',
               root + 'training/7/segm.nii.gz',
               root + 'training/14/segm.nii.gz',
               root + 'training/5/segm.nii.gz',
               root + 'training/070/segm.nii.gz']

    return T1path, FLAIRpath, IRpath, segpath


def data_val(root="./"):
    """
    Return:
    T1_val - Path to registered Bias Normalized T1
    FLAIR_val - Path to registered Bias Normalized FLAIR
    IR_val - Path to registered Bias Normalized IR
    segm_val - Path to segmentations
    """

    T1_val = root + 'training/148/pre/reg_T1.nii.gz'
    FLAIR_val = root + 'training/148/pre/FLAIR.nii.gz'
    IR_val = root + 'training/148/pre/reg_IR.nii.gz'
    segm_val = root + 'training/148/segm.nii.gz'

    return T1_val, FLAIR_val, IR_val, segm_val


def read_batch(in_dir, save_all=True):
    '''
    :param in_dir: directory where the .dcm files are stored
    :param save_all: True if we want to store the nii files to the same origin dir
    :return: array of .nii images
    '''
    reader = sitk.ImageSeriesReader()

    ids = reader.GetGDCMSeriesIDs(in_dir)

    images = []

    for i in ids:
        dicom_names = reader.GetGDCMSeriesFileNames(in_dir, i)
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        images.append(image)
        if save_all == True:
            series_name = reader.GetMetaData(0, '0008|103e')
            sitk.WriteImage(image, '{0}{1}.nii'.format(in_dir, series_name))

    # dicom_names = reader.GetGDCMSeriesFileNames(in_dir)

    return images


def hammers_train_partition_generator(ii, jj):
    return ['a{:02d}-pre.nii.gz'.format(i) for i in range(ii, jj)]


def hammers_val_partition_generator(ii, jj):
    return ['a{:02d}-pre.nii.gz'.format(i) for i in range(ii, jj)]


def hammers_outputs_generator(ii, jj, label):
    return {
        'a{:02d}-pre.nii.gz'.format(i): 'a{:02d}-{}.nii.gz'.format(i, label) for i in range(ii, jj)
    }





def create_hammers_partitions():
    partition = {
        'train': hammers_train_partition_generator(1, 19),
        'validation': hammers_val_partition_generator(19, 25)
    }

    outputs = hammers_outputs_generator(1, 25, 'cerebellum')

    return partition, outputs


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_ids, outputs, batch_size=32, dim=(128, 112, 80), n_channels=1, shuffle=True, root='.'):
        self.dim = dim
        self.batch_size = batch_size
        self.outputs = outputs
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.root = root
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):

        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        list_ids_temp = [self.list_ids[k] for k in indexes]

        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def __data_generation(self, list_ids_temp):
        X = []
        y = []

        for i, ID in enumerate(list_ids_temp):
            X.append(to_uint8(get_data(self.root + 'reduced/' + ID))[None, ...])
            y.append(to_uint8(get_data(self.root + 'reduced/' + self.outputs[ID]))[None, ...])

        X = np.array(X)
        y = np.array(y)

        return X, y



def step_decay(epoch, lr):
    #initial_rate = 0.1
    drop = 0.1
    epochs_drop = 10
    lrate = lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

