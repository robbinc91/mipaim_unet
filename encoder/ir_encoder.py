from keras.layers import Conv3D, MaxPool3D


def ir_encoder(img_input_IR, IMAGE_ORDERING):
    """
    :param img_input_IR: keras.layers.Input
    :param IMAGE_ORDERING: string
    :return:  Maxpool3D, Conv3D, Conv3D
    """

    # IR encoder

    conv_1_IR = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu',
                       name="CONV3D_1_IR", dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(img_input_IR)
    conv_11_IR = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu',
                        name="CONV3D_11_IR", dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_1_IR)
    maxpool_1_IR = MaxPool3D(name="MAXPOOL3D_1_IR",
                             data_format=IMAGE_ORDERING)(conv_11_IR)
    conv_2_IR = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu',
                       name="CONV3D_2_IR", dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(maxpool_1_IR)
    conv_21_IR = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu',
                        name="CONV3D_21_IR", dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_2_IR)
    maxpool_2_IR = MaxPool3D(name="MAXPOOL3D_2_IR",
                             data_format=IMAGE_ORDERING)(conv_21_IR)
    conv_3_IR = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu',
                       name="CONV3D_3_IR", dilation_rate=(1, 1, 1), data_format=IMAGE_ORDERING)(maxpool_2_IR)
    conv_31_IR = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu',
                        name="CONV3D_31_IR", dilation_rate=(2, 2, 2), data_format=IMAGE_ORDERING)(conv_3_IR)
    conv_32_IR = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu',
                        name="CONV3D_32_IR", dilation_rate=(3, 3, 3), data_format=IMAGE_ORDERING)(conv_31_IR)
    maxpool_3_IR = MaxPool3D(name="MAXPOOL3D_3_IR",
                             data_format=IMAGE_ORDERING)(conv_32_IR)

    return maxpool_3_IR, conv_21_IR, conv_32_IR
