from keras.layers import Conv3D, Conv2D, Softmax
from keras_contrib.layers import InstanceNormalization
#from keras.activations import softmax
#import tensorflow as tf

def output_mapper(_input,
                  num_labels=1,
                  activation='relu',
                  IMAGE_ORDERING='channels_first',
                  instance_normalization=False):

    Conv = Conv3D if len(_input.shape) == 5 else Conv2D

    _output = Conv(filters=num_labels,
                   kernel_size=1,
                   activation=activation,
                   strides=1,
                   padding='same',
                   data_format=IMAGE_ORDERING)(_input)
    
    normalization_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1

    if instance_normalization is True:
        _output = InstanceNormalization(dtype='float32')(_output)

    _output = Softmax(axis=1, dtype='float32')(_output)
    #_output = Activation('softmax', axis=1, dtype='float32')(_output)

    #_output = softmax(tf.cast(_output, 'float32'), axis=1)#(_output)

    return _output
