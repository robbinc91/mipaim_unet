from keras.layers import Conv3D, MaxPool3D


def base_encoder(img_input, IMAGE_ORDERING, ENCODE_NAME='T1'):
    """
    :param ENCODE_NAME: name of the encoding: 'T1', 'FLAIR', 'IR'
    :param img_input: keras.layers.Input
    :param IMAGE_ORDERING: string
    :return:  Maxpool3D, Conv3D, Conv3D
    """

    # FLAIR encoder

    conv_1 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_1_" + ENCODE_NAME, dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(img_input)
    conv_11 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_11_" + ENCODE_NAME, dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_1)
    maxpool_1 = MaxPool3D(name="MAXPOOL3D_1_" + ENCODE_NAME, data_format=IMAGE_ORDERING)(conv_11)
    conv_2 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_2_" + ENCODE_NAME, dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(maxpool_1)
    conv_21 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_21_" + ENCODE_NAME, dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_2)
    maxpool_2 = MaxPool3D(name="MAXPOOL3D_2_" + ENCODE_NAME, data_format=IMAGE_ORDERING)(conv_21)
    conv_3 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_3_" + ENCODE_NAME, dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(maxpool_2)
    conv_31 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_31_" + ENCODE_NAME, dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_3)
    conv_32 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_32_" + ENCODE_NAME, dilation_rate=(3, 3, 3), data_format=IMAGE_ORDERING)(conv_31)
    maxpool_3 = MaxPool3D(name="MAXPOOL3D_3_" + ENCODE_NAME, data_format=IMAGE_ORDERING)(conv_32)

    return maxpool_3, conv_21, conv_32
