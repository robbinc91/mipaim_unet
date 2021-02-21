from keras.layers import Conv3D, MaxPool3D


def t1_encoder(img_input_t1, IMAGE_ORDERING):

    """
    :param img_input_t1: keras.layers.Input
    :param IMAGE_ORDERING: string
    :return:  Maxpool3D, Conv3D, Conv3D
    """

    # T1 encoder

    conv_1_t1 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu', name='CONV3D_1_T1', dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(img_input_t1)
    conv_11_t1 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu', name='CONV3D_11_T1', dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_1_t1)
    maxpool_1_t1 = MaxPool3D(name='MAXPOOL3D_1_T1', data_format=IMAGE_ORDERING)(conv_11_t1)
    conv_2_t1 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_2_T1", dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(maxpool_1_t1)
    conv_21_t1 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_21_T1", dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_2_t1)
    maxpool_2_t1 = MaxPool3D(name="MAXPOOL3D_2_T1", data_format=IMAGE_ORDERING)(conv_21_t1)
    conv_3_t1 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_3_T1", dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(maxpool_2_t1)
    conv_31_t1 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_31_T1", dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_3_t1)
    conv_32_t1 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name="CONV3D_32_T1", dilation_rate=(3, 3, 3), data_format=IMAGE_ORDERING)(conv_31_t1)
    maxpool_3_t1 = MaxPool3D(name="MAXPOOL3D_3_T1", data_format=IMAGE_ORDERING)(conv_32_t1)

    return maxpool_3_t1, conv_21_t1, conv_32_t1
