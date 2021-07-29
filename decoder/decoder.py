from keras.layers import Conv3D, Conv3DTranspose, Concatenate, UpSampling3D, Dropout, Conv2D, Conv2DTranspose
from encoder import basic_rdim_inception, basic_naive_inception

def decode_inception(layers, naive=False, IMAGE_ORDERING='channels_first', dropout=None, only_3x3_filters=False):
    fn = basic_naive_inception if naive else basic_rdim_inception
    Conv = Conv3D if len(layers[0].shape) == 5 else Conv2D
    ConvTranspose = Conv3DTranspose if len(layers[0].shape) == 5 else Conv2DTranspose

    layer_6 = fn(layers[4], 64, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_6 =UpSampling3D(size=(3, 3, 3))(layer_6)
    layer_6 = ConvTranspose(filters=64, kernel_size=3, activation='relu', strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_6)

    layer_7 = Concatenate(axis=1)([layers[3], layer_6])
    layer_7 = fn(layer_7, 64, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_7 = UpSampling3D(size=(3, 3, 3))(layer_7)
    layer_7 = ConvTranspose(filters=64, kernel_size=3, activation='relu', strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_7)

    layer_8 = Concatenate(axis=1)([layers[2], layer_7])
    layer_8 = fn(layer_8, 32, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_8 = UpSampling3D(size=(3, 3, 3))(layer_8)
    layer_8 = ConvTranspose(filters=32, kernel_size=3, activation='relu', strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_8)

    layer_9 = Concatenate(axis=1)([layers[1], layer_8])
    layer_9 = fn(layer_9, 16, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)
    # layer_9 = UpSampling3D(size=(3, 3, 3))(layer_9)
    layer_9 = ConvTranspose(filters=16, kernel_size=3, activation='relu', strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_9)

    layer_10 = Concatenate(axis=1)([layers[0], layer_9])
    layer_10 = fn(layer_10, 8, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)

    output = Conv(filters=1, kernel_size=1, activation='relu', strides=1, padding='same', data_format=IMAGE_ORDERING)(layer_10)

    if dropout is not None:
        output = Dropout(dropout)(output)

    return output


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
        outputs_array.append(outputs['t1_output'])
        conv_21_array.append(conv_21['conv_21_t1'])
        conv_32_array.append(conv_32['conv_32_t1'])
    if inputs['flair_input'] is not None:
        inputs_array.append(inputs['flair_input'])
        outputs_array.append(outputs['flair_output'])
        conv_21_array.append(conv_21['conv_21_FLAIR'])
        conv_32_array.append(conv_32['conv_32_FLAIR'])
    if inputs['ir_input'] is not None:
        inputs_array.append(inputs['ir_input'])
        outputs_array.append(outputs['ir_output'])
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
