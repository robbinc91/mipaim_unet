from keras.models import Model
from keras.layers import Input
from keras.layers import Conv3D, MaxPool3D
from keras.layers import Conv3D, Conv3DTranspose, Concatenate

from encoder import encode, encode_inception
from decoder import decode, decode_inception


def vnet(t1, FLAIR=None, IR=None, IMAGE_ORDERING='channels_first', shape=(1, 240, 240, 48)):
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
    return Model(inputs, output)


def inception_unet(shape=(1, 240, 240, 48), IMAGE_ORDERING='channels_first'):
    input = Input(shape=shape)
    encoded_layers = encode_inception(input, False, IMAGE_ORDERING=IMAGE_ORDERING)
    output = decode_inception(encoded_layers, False, IMAGE_ORDERING=IMAGE_ORDERING)


    return Model(input, output)

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
