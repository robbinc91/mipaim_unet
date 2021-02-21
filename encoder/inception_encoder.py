from keras.layers import Conv3D, MaxPool3D, Concatenate


def basic_rdim_inception(img_input, nfilters=64, IMAGE_ORDERING='channels_first'):
    tower_1 = Conv3D(filters=nfilters, kernel_size=(1, 1, 1), padding='same', activation='relu', data_format=IMAGE_ORDERING)(img_input)

    tower_2 = Conv3D(filters=nfilters, kernel_size=(3, 3, 3), padding='same', activation='relu', data_format=IMAGE_ORDERING)(tower_1)

    tower_3 = Conv3D(filters=nfilters, kernel_size=(5, 5, 5), padding='same', activation='relu', data_format=IMAGE_ORDERING)(tower_1)

    tower_4 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format=IMAGE_ORDERING)(img_input)
    tower_4 = Conv3D(filters=nfilters, kernel_size=(1, 1, 1), padding='same', activation='relu', data_format=IMAGE_ORDERING)(tower_4)

    return Concatenate(axis=1)([tower_1, tower_2, tower_3, tower_4])


def basic_naive_inception(img_input, nfilters=64, IMAGE_ORDERING='channels_first'):
    tower_1 = Conv3D(filters=nfilters, kernel_size=(1, 1, 1), padding='same', activation='relu', data_format=IMAGE_ORDERING)(img_input)

    tower_2 = Conv3D(filters=nfilters, kernel_size=(3, 3, 3), padding='same', activation='relu', data_format=IMAGE_ORDERING)(img_input)

    tower_3 = Conv3D(filters=nfilters, kernel_size=(5, 5, 5), padding='same', activation='relu', data_format=IMAGE_ORDERING)(img_input)

    tower_4 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format=IMAGE_ORDERING)(img_input)

    return Concatenate([tower_1, tower_2, tower_3, tower_4])

def encode_inception(img_input, naive=False, IMAGE_ORDERING='channels_first'):
    fn = basic_naive_inception if naive else basic_rdim_inception

    layer_1 = fn(img_input, 32, IMAGE_ORDERING=IMAGE_ORDERING)

    layer_2 = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', data_format=IMAGE_ORDERING)(layer_1)
    layer_2 = fn(layer_2, 64, IMAGE_ORDERING=IMAGE_ORDERING)

    layer_3 = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', data_format=IMAGE_ORDERING)(layer_2)
    layer_3 = fn(layer_3, 128, IMAGE_ORDERING=IMAGE_ORDERING)

    layer_4 = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', data_format=IMAGE_ORDERING)(layer_3)
    layer_4 = fn(layer_4, 256, IMAGE_ORDERING=IMAGE_ORDERING)

    layer_5 = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', data_format=IMAGE_ORDERING)(layer_4)
    layer_5 = fn(layer_5, 512, IMAGE_ORDERING=IMAGE_ORDERING)

    return [layer_1, layer_2, layer_3, layer_4, layer_5]


