from keras.layers import Conv3D, Conv3DTranspose, Concatenate, \
    UpSampling3D, Dropout, Conv2D, Conv2DTranspose, Flatten, Dense, Activation, Add, Softmax


def output_mapper(_input, num_labels=1, activation='relu', IMAGE_ORDERING='channels_first'):

    Conv = Conv3D if len(_input.shape) == 5 else Conv2D

    _output = Conv(filters=num_labels,
                  kernel_size=1,
                  activation=activation,
                  strides=1,
                  padding='same',
                  data_format=IMAGE_ORDERING)(_input)

    #_output = Activation('softmax')(_output)

    return _output


