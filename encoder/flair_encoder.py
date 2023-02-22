from keras.layers import Conv3D, MaxPool3D


def flair_encoder(img_input_FLAIR, IMAGE_ORDERING):
    """
    :param img_input_FLAIR: keras.layers.Input
    :param IMAGE_ORDERING: string
    :return:  Maxpool3D, Conv3D, Conv3D
    """

    # FLAIR encoder

    conv_1_FLAIR = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu',
                          name="CONV3D_1_FLAIR", dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(img_input_FLAIR)
    conv_11_FLAIR = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu',
                           name="CONV3D_11_FLAIR", dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_1_FLAIR)
    maxpool_1_FLAIR = MaxPool3D(
        name="MAXPOOL3D_1_FLAIR", data_format=IMAGE_ORDERING)(conv_11_FLAIR)
    conv_2_FLAIR = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu',
                          name="CONV3D_2_FLAIR", dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(maxpool_1_FLAIR)
    conv_21_FLAIR = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu',
                           name="CONV3D_21_FLAIR", dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_2_FLAIR)
    maxpool_2_FLAIR = MaxPool3D(
        name="MAXPOOL3D_2_FLAIR", data_format=IMAGE_ORDERING)(conv_21_FLAIR)
    conv_3_FLAIR = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu',
                          name="CONV3D_3_FLAIR", dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(maxpool_2_FLAIR)
    conv_31_FLAIR = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu',
                           name="CONV3D_31_FLAIR", dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_3_FLAIR)
    conv_32_FLAIR = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu',
                           name="CONV3D_32_FLAIR", dilation_rate=(3, 3, 3), data_format=IMAGE_ORDERING)(conv_31_FLAIR)
    maxpool_3_FLAIR = MaxPool3D(
        name="MAXPOOL3D_3_FLAIR", data_format=IMAGE_ORDERING)(conv_32_FLAIR)

    return maxpool_3_FLAIR, conv_21_FLAIR, conv_32_FLAIR
