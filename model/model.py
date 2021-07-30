from keras.models import Model
from keras.layers import Input, Dropout
from keras.layers import Conv3D, MaxPool3D
from keras.layers import Conv3D, Conv3DTranspose, Concatenate

from encoder import encode, encode_inception
from decoder import decode, decode_inception, decode_classification


def unet(t1, FLAIR=None, IR=None, IMAGE_ORDERING='channels_first', shape=(1, 240, 240, 48)):
    """
    :param t1: any
    :param FLAIR: any | None
    :param IR: any | None
    :param IMAGE_ORDERING: string
    :param shape: tuple
    :return: keras.models.Model
    """

    # encoder stuff
    inputs, outputs, conv_21, conv_32, = encode(t1, FLAIR, IR, IMAGE_ORDERING, shape)

    # decoder stuff
    inputs, output = decode(inputs, outputs, conv_21, conv_32, IMAGE_ORDERING)
    return Model(inputs[0], output)


def inception_unet(shape=(1, 240, 240, 48), IMAGE_ORDERING='channels_first', only_3x3_filters=False, dropout=None):
    input = Input(shape=shape)
    encoded_layers = encode_inception(input, False, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    output = decode_inception(encoded_layers, False, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters, dropout=dropout)


    return Model(input, output)


def parcellation_inception_unet(shape=(1, 128, 80, 80), IMAGE_ORDERING='channels_first', only_3x3_filter=False, dropout=None, labels=28):
    _input = Input(shape=shape)
    _encoded_layers = []
    _outputs = []
    for i in range(labels):
        _encoded_layers.append(encode_inception(_input, False, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filter))
        _outputs.append(decode_inception(_encoded_layers[i], False, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filter, dropout=dropout))

    _output = Concatenate(axis=1)(_outputs)
    return Model(_input, _output)


def parcellation_inception_unet_reduced(shape=(1, 128, 80, 80),
                                        IMAGE_ORDERING='channels_first',
                                        only_3x3_filters=False,
                                        dropout=None,
                                        labels=28,
                                        activation='relu',
                                        final_droput=None):
    _input = Input(shape=shape)

    encoded_layers = encode_inception(_input, False, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    general_output = decode_inception(encoded_layers, False, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters,
                              dropout=dropout)

    _outputs = []
    for i in range(labels):
        _outputs.append(Conv3D(filters=1,
                               kernel_size=1,
                               activation=activation, strides=1, padding='same', data_format=IMAGE_ORDERING)(general_output))

        if final_droput is not None:
            _outputs[i] = Dropout(final_droput)(_outputs[i])

    _output = Concatenate(axis=1)(_outputs)
    return Model(_input, _output)


def classification_model(shape=(1, 128, 80, 80),
                         IMAGE_ORDERING='channels_first',
                         only_3x3_filters=False,
                         dropout=None):
    _input = Input(shape=shape)

    encoded_layers = encode_inception(_input, False, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    _decoded = decode_classification(encoded_layers[-1], dropout=dropout)

    return Model(_input, _decoded)


def model_thresholding():
    IMAGE_ORDERING = "channels_first"
    img_input = Input(shape=(1, 240, 240, 48))
    conv_1 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_1",
                    dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(img_input)
    maxpool_1 = MaxPool3D(name="MAXPOOL3D_1", data_format=IMAGE_ORDERING)(conv_1)
    conv_2 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_2",
                    dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(maxpool_1)
    maxpool_2 = MaxPool3D(name="MAXPOOL3D_2", data_format=IMAGE_ORDERING)(conv_2)
    conv_3 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_3",
                    dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(maxpool_2)

    convt_1 = Conv3DTranspose(16, kernel_size=(2, 2, 2), strides=(2, 2, 2), name="CONV3DT_1", activation='relu',
                              data_format=IMAGE_ORDERING)(conv_3)
    concat_1 = Concatenate(axis=1)([convt_1, conv_2])
    conv_4 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_4",
                    data_format=IMAGE_ORDERING)(concat_1)
    convt_2 = Conv3DTranspose(4, kernel_size=(2, 2, 2), strides=(2, 2, 2), name="CONV3DT_2", activation='relu',
                              data_format=IMAGE_ORDERING)(conv_4)
    concat_2 = Concatenate(axis=1)([convt_2, conv_1])
    conv_5 = Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same', activation='sigmoid', name="CONV3D_5",
                    data_format=IMAGE_ORDERING)(concat_2)
    return Model(img_input, conv_5)

    concat_2 = Concatenate(axis=1)([convt_2, conv_1])
    conv_5 = Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same', activation='sigmoid', name="CONV3D_5",
                    data_format=IMAGE_ORDERING)(concat_2)
    return Model(img_input, conv_5)
