import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
import keras
from utils.preprocess import to_uint8, get_data, histeq
import math
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

# Cerebellum segmentation
train_prt = [i for i in range(1, 19)] + [31, 34, 35, 36, 39, 43, 44]
dev_prt = [i for i in range(19, 25)] + [32, 37, 40]
test_prt = [i for i in range(25, 31)] + [33, 38, 41, 42]
train_prt_augm = [i for i in range(1, 250)]
dev_prt_augm = [i for i in range(250, 331)]

# Cerebellum parcellation
#cersegsys_train_prt = [31, 32, 33, 36, 37, 38, 42]
#cersegsys_dev_prt = [34, 39, 41]
#cersegsys_test_prt = [35, 40, 43, 44]

cersegsys_train_prt = [31, 32, 36, 37, 38, 39, 43, 44, 45, 46, 47, 50, 52, 54, 61, 62, 65]
cersegsys_dev_prt = [33, 66, 70]
cersegsys_test_prt = [i for i in range(
    31, 73) if i not in cersegsys_train_prt and i not in cersegsys_dev_prt]


#cersegsys_train_prt_augm = [i for i in range(1, 71)]
#cersegsys_dev_prt_augm = [i for i in range(71, 101)]


cersegsys_train_prt_augm = [i for i in range(1, 601)]  # Added new images
cersegsys_dev_prt_augm = [i for i in range(601, 761)]


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
    T1Path = [root + 'pre/' +
              'a{:02d}-reg-res.nii.gz'.format(i) for i in range(1, 25)]
    segpath = [root + 'pre/' +
               'a{:02d}-reg-seg-res.nii.gz'.format(i) for i in range(1, 25)]

    return T1Path, segpath


def hammers_2017_data_preprocessed_train(root):
    T1Path = [root + 'final/' +
              'a{:02d}-pre.nii.gz'.format(i) for i in range(1, 25)]
    segpath = [root + 'final/' +
               'a{:02d}-cerebellum.nii.gz'.format(i) for i in range(1, 25)]

    return T1Path, segpath


def hammers_2017_data_preprocessed_train_reduced(root, label='cerebellum'):
    T1Path = [root + 'reduced/' +
              'a{:02d}-pre.nii.gz'.format(i) for i in range(1, 25)]
    segpath = [root + 'reduced/' +
               'a{:02d}-{}.nii.gz'.format(i, label) for i in range(1, 25)]

    return T1Path, segpath


def mrbrains2018_data_val(root="./", t1=True, flair=False, ir=False):

    T1_val = root + 'training/148/pre/reg_T1.nii.gz'
    segm_val = root + 'training/148/segm.nii.gz'

    return T1_val, segm_val


def hammers_2017_data_val(root):
    T1_val = [root + 'pre/' +
              'a{:02d}-reg-res.nii.gz'.format(i) for i in range(19, 25)]
    segm_val = [root + 'pre/' +
                'a{:02d}-reg-seg-res.nii.gz'.format(i) for i in range(19, 25)]

    return T1_val, segm_val


def hammers_2017_data_evaluation(root):
    T1_val = [root + 'pre/' +
              'a{:02d}-reg-res.nii.gz'.format(i) for i in range(25, 31)]
    segm_eval = [root + 'pre/' +
                 'a{:02d}-reg-seg-res.nii.gz'.format(i) for i in range(25, 31)]

    return T1_val, segm_eval


def hammers_2017_data_evaluation_reduced(root, label='cerebellum'):
    T1_val = [root + 'reduced/' +
              'a{:02d}-pre.nii.gz'.format(i) for i in range(25, 31)]
    segm_eval = [root + 'reduced/' +
                 'a{:02d}-{}.nii.gz'.format(i, label) for i in range(25, 31)]

    return T1_val, segm_eval


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


def hammers_partition_generator(rng, rnga=None):
    if rnga is None:
        return ['a{:02d}-pre.nii.gz'.format(i) for i in rng]
    return ['a{:02d}-pre.nii.gz'.format(i) for i in rng] + ['a{:03d}-pre.nii.gz'.format(i) for i in rnga]


def _hammers_output_generator(rng, rnga=None, label='cerebellum'):
    if rnga is None:
        return {
            'a{:02d}-pre.nii.gz'.format(i): 'a{:02d}-{}.nii.gz'.format(i, label) for i in rng
        }

    ret = {
        'a{:02d}-pre.nii.gz'.format(i): 'a{:02d}-{}.nii.gz'.format(i, label) for i in rng
    }
    ret.update({
        'a{:03d}-pre.nii.gz'.format(i): 'a{:03d}-{}.nii.gz'.format(i, label) for i in rnga
    })
    return ret


def create_hammers_partitions_new(label='cerebellum', use_augmentation=False):
    if use_augmentation is False:
        partition = {
            'train': hammers_partition_generator(train_prt, None),
            'validation': hammers_partition_generator(dev_prt, None)
        }
        outputs = _hammers_output_generator(train_prt + dev_prt, None, label)
    else:
        partition = {
            'train': hammers_partition_generator(train_prt, train_prt_augm),
            'validation': hammers_partition_generator(dev_prt, dev_prt_augm)
        }
        outputs = _hammers_output_generator(
            train_prt + dev_prt, train_prt_augm + dev_prt_augm, label)

    return partition, outputs


def cersegsys_partition_generator(rng, rnga=None, second_lbl=''):
    if rnga is None:
        return ['a{:02d}{}.nii.gz'.format(i, second_lbl) for i in rng]
    return ['a{:02d}{}.nii.gz'.format(i, second_lbl) for i in rng] + ['a{:03d}{}.nii.gz'.format(i, second_lbl) for i in rnga]


def cersegsys_output_generator(rng, rnga=None, label='cerebellum', second_lbl=''):
    if rnga is None:
        return {
            'a{:02d}{}.nii.gz'.format(i, second_lbl): 'a{:02d}-{}.nii.gz'.format(i, label) for i in rng
        }

    ret = {
        'a{:02d}{}.nii.gz'.format(i, second_lbl): 'a{:02d}-{}.nii.gz'.format(i, label) for i in rng
    }
    ret.update({
        'a{:03d}{}.nii.gz'.format(i, second_lbl): 'a{:03d}-{}.nii.gz'.format(i, label) for i in rnga
    })

    return ret


def create_cersegsys_partitions(label='cerebellum', use_augmentation=False, second_lbl='', use_originals=True):
    if use_originals:
        if use_augmentation is False:
            partition = {
                'train': cersegsys_partition_generator(cersegsys_train_prt, None, second_lbl),
                'validation': cersegsys_partition_generator(cersegsys_dev_prt, None, second_lbl)
            }
            outputs = cersegsys_output_generator(
                cersegsys_train_prt + cersegsys_dev_prt, None, label, second_lbl)
        else:
            partition = {
                'train': cersegsys_partition_generator(cersegsys_train_prt, cersegsys_train_prt_augm, second_lbl),
                'validation': cersegsys_partition_generator(cersegsys_dev_prt, cersegsys_dev_prt_augm, second_lbl)
            }
            outputs = cersegsys_output_generator(
                cersegsys_train_prt + cersegsys_dev_prt, cersegsys_train_prt_augm + cersegsys_dev_prt_augm, label, second_lbl)
    elif use_augmentation:
        partition = {
            'train': cersegsys_partition_generator([], cersegsys_train_prt_augm, second_lbl),
            'validation': cersegsys_partition_generator([], cersegsys_dev_prt_augm, second_lbl)
        }
        outputs = cersegsys_output_generator(
            [], cersegsys_train_prt_augm + cersegsys_dev_prt_augm, label, second_lbl)

    return partition, outputs


THALAMIC_IMAGES = sorted([98, 66, 85, 109, 40, 61, 91, 52, 58, 87, 68, 93, 93, 19, 70, 44, 12, 70, 37, 108, 92, 57, 79, 108, 61, 43, 61, 50, 67, 56, 93, 15, 42, 37, 101, 39, 91, 58, 85, 92, 92, 36, 43, 36, 47, 76, 35, 88, 81, 27, 65, 70, 62, 61, 94, 8, 2, 30, 95, 103, 16, 54, 58, 16, 109, 71, 52, 22, 43, 46])
TOTAL_AUGMENTED_THALAMIC_ = 70 * 15
def create_thalamic_partitions(label='-seg', use_augmentation=False, second_lbl='', use_originals=True, train_percent=90, label_append=''):
    import random
    originals = random.sample(THALAMIC_IMAGES, k=63)
    augmented = random.sample([i for i in range(1, TOTAL_AUGMENTED_THALAMIC_ + 1)], k = 945)

    outputs = {
        '{:02d}_cropped_48x64x64'.format(i) + label_append + '.nii.gz': '{:02d}-seg_cropped_48x64x64.nii.gz'.format(i)
        for i in THALAMIC_IMAGES
    }
    outputs.update({
        'augmented/{:03d}'.format(i) + label_append + '.nii.gz': 'augmented/{:03d}-seg.nii.gz'.format(i)
        for i in range(1, TOTAL_AUGMENTED_THALAMIC_ + 1)
    })


    train_partition = ['{:02d}_cropped_48x64x64'.format(i) + label_append + '.nii.gz' for i in originals] + ['augmented/{:03d}'.format(j) + label_append + '.nii.gz' for j in augmented]
    val_partition = ['{:02d}_cropped_48x64x64'.format(i) + label_append + '.nii.gz' for i in THALAMIC_IMAGES if i not in originals] + ['augmented/{:03d}'.format(j) + label_append + '.nii.gz' for j in range(1, TOTAL_AUGMENTED_THALAMIC_ + 1) if j not in augmented]

    partition = {
        'train': train_partition,
        'validation': val_partition
    }

    return partition, outputs


def create_hammers_partitions(label='cerebellum'):
    partition = {
        'train': hammers_train_partition_generator(1, 19),
        'validation': hammers_val_partition_generator(19, 25)
    }

    outputs = hammers_outputs_generator(1, 25, label)

    return partition, outputs


def onehot(label_image, num_labels=None):
    """Performs one-hot encoding.

    Args:
        label_image (numpy.ndarray): The label image to convert to be encoded.
        num_labels (int): The total number of labels. If ``None``, it will be
            calcualted as the number of unique values of input ``label_image``.

    Returns:
        numpy.ndarray: The encoded label image. The first dimension is channel.

    """
    if num_labels is None:
        num_labels = len(np.unique(label_image)) - 1  # without background 0
    result = np.zeros((num_labels + 1, label_image.size), dtype=bool)
    result[label_image.flatten(), np.arange(label_image.size)] = 1
    result = result.reshape(-1, *label_image.shape)
    return result


class DataGenerator(keras.utils.Sequence):

    def __init__(self,
                 list_ids,
                 outputs,
                 batch_size=32,
                 dim=(128, 112, 80),
                 n_channels=1,
                 shuffle=True,
                 root='.',
                 histogram_equalization=False,
                 in_folder='preprocessed-full',
                 binary=False,
                 labels=None,
                 filter_label=None,
                 is_segmentation=True,
                 input_mask_prefix=None):
        self.dim = dim
        self.batch_size = batch_size
        self.outputs = outputs
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.root = root
        self.histogram_equalization = histogram_equalization
        self.in_folder = in_folder
        self.binary = binary
        self.labels = labels
        self.filter_label = filter_label
        self.is_segmentation = is_segmentation
        self.classes = {}
        self.input_mask_prefix = input_mask_prefix
        if not self.is_segmentation:
            with open('{0}{1}'.format(self.root, 'classes.txt')) as input_file:
                for line in input_file:
                    line_ = line.split()
                    if len(line_) == 2:
                        self.classes[line_[0]] = int(line_[1]) - 1
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):

        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]

        list_ids_temp = [self.list_ids[k] for k in indexes]

        return self.__data_generation(list_ids_temp)

        #return X, y

    def __data_generation(self, list_ids_temp):
        X = []
        y = []
        X1 = []

        for i, ID in enumerate(list_ids_temp):
            x = get_data(self.root + self.in_folder + '/' + ID)#.astype(np.float32)
            if self.histogram_equalization is True:
                x = x.round().astype(np.uint8)
                x = histeq(x)

            X.append(x[None, ...])

            if self.input_mask_prefix is not None:
                #print('opening', (self.root + self.in_folder + '/' + ID).replace('.nii', f'{self.input_mask_prefix}.nii'))
                x_ = get_data((self.root + self.in_folder + '/' + ID).replace('.nii', f'{self.input_mask_prefix}.nii'))
                X1.append(x_[None, ...])
            

            if self.is_segmentation:
                # it's a segmentation/parcellation task

                if (self.labels is not None) and (self.binary is True):
                    # binary parcellation
                    yLabels = list()
                    _data = get_data(self.root + self.in_folder +
                                     '/' + self.outputs[ID]).round().astype(int)
                    for label_num in self.labels:
                        # if label_num == 12:
                        #    continue
                        #yLabels.append(np.array(_data == label_num).astype(int)[None, ...])
                        yLabels.append(
                            np.array(_data == label_num).astype(int))
                        #yLabels.append(to_uint8(np.array(_data == label_num))[None, ...])
                        #yLabels.append(to_uint8(np.array(_data == label_num)))

                    y.append(yLabels)
                    # y.append(_data)

                elif self.filter_label is not None:
                    # binary parcellation, filtering by label. too slow
                    _data = get_data(self.root + self.in_folder +
                                     '/' + self.outputs[ID]).round().astype(int)
                    y.append(
                        to_uint8(np.array(_data == self.filter_label))[None, ...])
                else:
                    # segmentation, or non binary parcellation
                    tmp = get_data(self.root + self.in_folder +
                                   '/' + self.outputs[ID]).round().astype(np.uint8)  # [None, ...]
                    #_img = []
                    # for i in range(1, 5):
                    #    _img.append(tmp == i)

                    # _img = np.array(_img).astype(np.float32)#[None, ...]
                    #one_hot = OneHotEncoder()
                    #_img =  np.stack([tmp==i for i in range(1, 5)], axis=0).astype(np.float32)
                    _img = onehot(tmp, 13)
                    # TODO: make this automatically
                    # 3 brainstem parcellation
                    # 7 cerebellum parcellation
                    # 2 cerebellum tissues
                    # 13 thalamic nuclei

                    #print(ID, self.outputs[ID], '_img.shape:', _img.shape)
                    #_img = tf.transpose(_img, perm=[3, 0, 1, 2])
                    # print(_img.shape)
                    y.append(_img.astype(np.float32))
                    # TODO: check this
                    # if self.binary:
                    #    #print('----------------------------'*10)
                    #    #y.append(to_uint8(get_data(self.root + self.in_folder + '/' + self.outputs[ID])))
                    #    y.append(get_data(self.root + self.in_folder + '/' + self.outputs[ID]))
                    # else:
                    #    y.append(get_data(self.root + self.in_folder + '/' + self.outputs[ID]).round().astype(int)[None, ...])
            else:
                # it's a classification task
                # return the item's classes (no multiclass items)
                y.append(self.classes[ID])

        X = np.array(X)
        X1 = np.array(X1)

        if not self.is_segmentation:
            y = to_categorical(y, num_classes=3)
            #y = np.array(y)
            # pass
        else:
            y = np.array(y)

        if self.input_mask_prefix is not None:
            return {'input_1': X, 'input_2': X1}, y
        
        return X, y


def step_decay(epoch, lr):
    #initial_rate = 0.1
    drop = 0.1
    epochs_drop = 10
    lrate = lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate
