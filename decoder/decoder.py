from keras.layers import Conv3D, Conv3DTranspose, Concatenate, \
    UpSampling3D, Dropout, Conv2D, Conv2DTranspose, Flatten, Dense, Activation, Add, Softmax
  
from keras.layers.normalization import BatchNormalization
from encoder import basic_rdim_inception, basic_naive_inception
from keras_contrib.layers import InstanceNormalization

def decode_parcellation(layers,
                        naive=False,
                        IMAGE_ORDERING='channels_first',
                        dropout=None,
                        only_3x3_filters=False,
                        filters_dim=None,
                        num_labels=28,
                        instance_normalization=False):

    if filters_dim is None:
        filters_dim = [8, 16, 32, 64, 128]

    fn = basic_naive_inception if naive else basic_rdim_inception
    Conv = Conv3D if len(layers[0].shape) == 5 else Conv2D
    ConvTranspose = Conv3DTranspose if len(layers[0].shape) == 5 else Conv2DTranspose

    layer_51 = Conv(filters=num_labels,
                    kernel_size=1,
                    activation='relu',
                    strides=1,
                    padding='same',
                    data_format=IMAGE_ORDERING)(layers[4])
    layer_51 = ConvTranspose(filters=num_labels,
                             kernel_size=3,
                             activation='relu',
                             strides=2,
                             padding='same',
                             data_format=IMAGE_ORDERING)(layer_51)

    layer_6 = fn(layers[4], filters_dim[4], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_6 =UpSampling3D(size=(3, 3, 3))(layer_6)
    layer_6 = ConvTranspose(filters=filters_dim[4],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING)(layer_6)

    normalization_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1

    if instance_normalization is True:
        layer_6 = InstanceNormalization(dtype='float32')(layer_6)

    layer_61 = Conv(filters=num_labels,
                   kernel_size=1,
                   activation='relu',
                   strides=1,
                   padding='same',
                   data_format=IMAGE_ORDERING)(layer_6)
    layer_61 = Add()([layer_51, layer_61])
    layer_61 = ConvTranspose(filters=num_labels,
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING)(layer_61)


    layer_7 = Concatenate(axis=1)([layers[3], layer_6])
    layer_7 = fn(layer_7, filters_dim[3], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_7 = UpSampling3D(size=(3, 3, 3))(layer_7)
    layer_7 = ConvTranspose(filters=filters_dim[3],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING)(layer_7)

    if instance_normalization is True:
        layer_7 = InstanceNormalization(dtype='float32')(layer_7)

    layer_71 = Conv(filters=num_labels,
                    kernel_size=1,
                    activation='relu',
                    strides=1,
                    padding='same',
                    data_format=IMAGE_ORDERING)(layer_7)
    layer_71 = Add()([layer_61, layer_71])
    layer_71 = ConvTranspose(filters=num_labels,
                             kernel_size=3,
                             activation='relu',
                             strides=2,
                             padding='same',
                             data_format=IMAGE_ORDERING)(layer_71)


    layer_8 = Concatenate(axis=1)([layers[2], layer_7])
    layer_8 = fn(layer_8, filters_dim[2], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_8 = UpSampling3D(size=(3, 3, 3))(layer_8)
    layer_8 = ConvTranspose(filters=filters_dim[2],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING)(layer_8)

    if instance_normalization is True:
        layer_8 = InstanceNormalization(dtype='float32')(layer_8)

    layer_81 = Conv(filters=num_labels,
                    kernel_size=1,
                    activation='relu',
                    strides=1,
                    padding='same',
                    data_format=IMAGE_ORDERING)(layer_8)
    layer_81 = Add()([layer_71, layer_81])
    layer_81 = ConvTranspose(filters=num_labels,
                             kernel_size=3,
                             activation='relu',
                             strides=2,
                             padding='same',
                             data_format=IMAGE_ORDERING)(layer_81)

    layer_9 = Concatenate(axis=1)([layers[1], layer_8])
    layer_9 = fn(layer_9, filters_dim[1], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_9 = UpSampling3D(size=(3, 3, 3))(layer_9)
    layer_9 = ConvTranspose(filters=filters_dim[1],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING)(layer_9)

    if instance_normalization is True:
        layer_9 = InstanceNormalization(dtype='float32')(layer_9)

    layer_91 = Conv(filters=num_labels,
                    kernel_size=1,
                    activation='relu',
                    strides=1,
                    padding='same',
                    data_format=IMAGE_ORDERING)(layer_9)
    layer_91 = Add()([layer_81, layer_91])
    #layer_91 = ConvTranspose(filters=filters_dim[0],
    #                         kernel_size=3,
    #                         activation='relu',
    #                         strides=2,
    #                         padding='same',
    #                         data_format=IMAGE_ORDERING)(layer_91)

    layer_10 = Concatenate(axis=1)([layers[0], layer_9])
    layer_10 = fn(layer_10, filters_dim[0], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)

    if instance_normalization is True:
        layer_10 = InstanceNormalization(dtype='float32')(layer_10)

    _decoded = Conv(filters=num_labels,
                   kernel_size=3,
                   activation='relu',
                   strides=1,
                   padding='same',
                   data_format=IMAGE_ORDERING)(layer_10)

    # print(layer_91.shape)
    # print(_decoded.shape)
    _decoded = Add()([layer_91, _decoded])

    _decoded = Conv(filters=num_labels,
                   kernel_size=1,
                   activation='relu',
                   padding='valid',
                   data_format=IMAGE_ORDERING)(_decoded)

    if dropout is not None:
        _decoded = Dropout(dropout)(_decoded)


    
    _decoded = Activation('softmax')(_decoded)

    return _decoded


def decode_inception(layers,
                     naive=False,
                     IMAGE_ORDERING='channels_first',
                     dropout=None,
                     only_3x3_filters=False,
                     filters_dim=None,
                     instance_normalization=False):

    if filters_dim is None:
        filters_dim = [8, 16, 32, 64, 128]

    fn = basic_naive_inception if naive else basic_rdim_inception
    Conv = Conv3D if len(layers[0].shape) == 5 else Conv2D
    ConvTranspose = Conv3DTranspose if len(layers[0].shape) == 5 else Conv2DTranspose

    layer_6 = fn(layers[4], filters_dim[4], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_6 =UpSampling3D(size=(3, 3, 3))(layer_6)
    layer_6 = ConvTranspose(filters=filters_dim[4],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING)(layer_6)

    normalization_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1

    if instance_normalization is True:
        layer_6 = InstanceNormalization(dtype='float32')(layer_6)

    layer_7 = Concatenate(axis=1)([layers[3], layer_6])
    layer_7 = fn(layer_7, filters_dim[3], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_7 = UpSampling3D(size=(3, 3, 3))(layer_7)
    layer_7 = ConvTranspose(filters=filters_dim[3],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING)(layer_7)

    if instance_normalization is True:
        layer_7 = InstanceNormalization(dtype='float32')(layer_7)

    layer_8 = Concatenate(axis=1)([layers[2], layer_7])
    layer_8 = fn(layer_8, filters_dim[2], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_8 = UpSampling3D(size=(3, 3, 3))(layer_8)
    layer_8 = ConvTranspose(filters=filters_dim[2],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING)(layer_8)

    if instance_normalization is True:
        layer_8 = InstanceNormalization(dtype='float32')(layer_8)

    layer_9 = Concatenate(axis=1)([layers[1], layer_8])
    layer_9 = fn(layer_9, filters_dim[1], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_9 = UpSampling3D(size=(3, 3, 3))(layer_9)
    layer_9 = ConvTranspose(filters=filters_dim[1],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING)(layer_9)

    if instance_normalization is True:
        layer_9 = InstanceNormalization(dtype='float32')(layer_9)

    layer_10 = Concatenate(axis=1)([layers[0], layer_9])
    layer_10 = fn(layer_10, filters_dim[0], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)

    if instance_normalization is True:
        layer_10 = InstanceNormalization(dtype='float32')(layer_10)

    
    _decoded = Conv(filters=1,
                  kernel_size=1,
                  activation='relu',
                  strides=1,
                  padding='same',
                  data_format=IMAGE_ORDERING)(layer_10)

    if dropout is not None:
        _decoded = Dropout(dropout)(_decoded)


    _decoded = Activation('softmax')(_decoded)

    

    return _decoded


def decode_inception_v2(layers,
                     naive=False,
                     IMAGE_ORDERING='channels_first',
                     dropout=None,
                     only_3x3_filters=False,
                     filters_dim=None,
                     instance_normalization=False,
                     kernel_initializer=None):

    if filters_dim is None:
        filters_dim = [8, 16, 32, 64, 128]

    fn = basic_naive_inception if naive else basic_rdim_inception
    ConvTranspose = Conv3DTranspose if len(layers[0].shape) == 5 else Conv2DTranspose

    normalization_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1

    _decoded = fn(layers[4], filters_dim[4], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)

    _decoded = ConvTranspose(filters=filters_dim[4],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING,
                            kernel_initializer=kernel_initializer)(_decoded)

    if instance_normalization is True:
        _decoded = InstanceNormalization(dtype='float32', name="decoder_instance_normalization_1")(_decoded)

    _decoded = Concatenate(axis=1, name='decoder_concatenate_1')([layers[3], _decoded])
    _decoded = fn(_decoded, filters_dim[3], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters, kernel_initializer=kernel_initializer)

    _decoded = ConvTranspose(filters=filters_dim[3],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING,
                            kernel_initializer=kernel_initializer)(_decoded)

    if instance_normalization is True:
        _decoded = InstanceNormalization(dtype='float32', name="decoder_instance_normalization_2")(_decoded)

    #print('concatenating shapes:', _decoded.shape, layers[2].shape)
    _decoded = Concatenate(axis=1, name='decoder_concatenate_2')([layers[2], _decoded])
    _decoded = fn(_decoded, filters_dim[2], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters, kernel_initializer=kernel_initializer)

    _decoded = ConvTranspose(filters=filters_dim[2],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING,
                            kernel_initializer=kernel_initializer)(_decoded)

    if instance_normalization is True:
        _decoded = InstanceNormalization(dtype='float32', name="decoder_instance_normalization_3")(_decoded)


    _decoded = Concatenate(axis=1, name='decoder_concatenate_3')([layers[1], _decoded])
    _decoded = fn(_decoded, filters_dim[1], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters, kernel_initializer=kernel_initializer)

    _decoded = ConvTranspose(filters=filters_dim[1],
                            kernel_size=3,
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format=IMAGE_ORDERING,
                            kernel_initializer=kernel_initializer)(_decoded)

    if instance_normalization is True:
        _decoded = InstanceNormalization(dtype='float32', name="decoder_instance_normalization_4")(_decoded)

    _decoded = Concatenate(axis=1, name='decoder_concatenate_4')([layers[0], _decoded])

    _decoded = fn(_decoded, filters_dim[0], IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters, kernel_initializer=kernel_initializer)

    if dropout is not None:
        _decoded = Dropout(dropout)(_decoded)

    return _decoded


def decode_classification(_layers, dropout=None, n_classes=3):
    _decoded = Flatten()(_layers)
    _decoded = Dense(256)(_decoded)
    _decoded = Activation('relu')(_decoded)
    if dropout is not None:
        _decoded = Dropout(dropout)(_decoded)

    _decoded = Dense(256, activation='relu')(_decoded)
    _decoded = Dense(n_classes, activation='softmax')(_decoded)

    return _decoded


def decode(inputs, outputs, conv_21, conv_32, IMAGE_ORDERING='channels_first'):

    """
    :param inputs: dict
    :param outputs: dict
    :param conv_21: dict
    :param conv_32: dict
    :param IMAGE_ORDERING: string
    :return: array, Conv3D
    """

    inputs_array = []
    outputs_array = []
    conv_21_array = []
    conv_32_array = []

    if inputs['t1_input'] is not None:
        inputs_array.append(inputs['t1_input'])
        outputs_array.append(outputs['t1_decoded'])
        conv_21_array.append(conv_21['conv_21_t1'])
        conv_32_array.append(conv_32['conv_32_t1'])
    if inputs['flair_input'] is not None:
        inputs_array.append(inputs['flair_input'])
        outputs_array.append(outputs['flair_decoded'])
        conv_21_array.append(conv_21['conv_21_FLAIR'])
        conv_32_array.append(conv_32['conv_32_FLAIR'])
    if inputs['ir_input'] is not None:
        inputs_array.append(inputs['ir_input'])
        outputs_array.append(outputs['ir_decoded'])
        conv_21_array.append(conv_21['conv_21_IR'])
        conv_32_array.append(conv_32['conv_32_IR'])

    ## concatenate
    concat_all = Concatenate(axis=1)(outputs_array) if len(outputs_array) > 1 else outputs_array[0]
    conv_all = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu', name='CONV3D_all', data_format=IMAGE_ORDERING)(concat_all)
    conv_all = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu', name='CONV3D_all_2',
                      data_format=IMAGE_ORDERING)(conv_all)

    ## decode
    convt_1 = Conv3DTranspose(32, kernel_size=(2, 2, 2), strides=(2, 2, 2), name="CONV3DT_1", activation='relu', data_format=IMAGE_ORDERING)(conv_all)
    concat_1 = Concatenate(axis=1)([convt_1] + conv_32_array)
    conv_4 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_4", data_format=IMAGE_ORDERING)(concat_1)
    conv_41 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_41", data_format=IMAGE_ORDERING)(conv_4)
    convt_2 = Conv3DTranspose(16, kernel_size=(2, 2, 2), strides=(2, 2, 2), name="CONV3DT_2", activation='relu', data_format=IMAGE_ORDERING)(conv_41)
    concat_2 = Concatenate(axis=1)([convt_2] + conv_21_array)
    conv_5 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_5",  data_format=IMAGE_ORDERING)(concat_2)
    conv_51 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_51", data_format=IMAGE_ORDERING)(conv_5)
    convt_3 = Conv3DTranspose(4, kernel_size=(2, 2, 2), strides=(2, 2, 2), name="CONV3DT_3", activation='relu', data_format=IMAGE_ORDERING)(conv_51)
    conv_6 = Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_6", data_format=IMAGE_ORDERING)(convt_3)

    return inputs_array, conv_6
