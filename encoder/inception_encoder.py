from keras.layers import Conv3D, MaxPool3D, Concatenate, MaxPool2D, Conv2D


def basic_rdim_inception(img_input, nfilters=64, IMAGE_ORDERING='channels_first', only_3x3_filters=False):
    Conv = Conv3D if len(img_input.shape) == 5 else Conv2D
    MaxPool = MaxPool3D if len(img_input.shape) == 5 else MaxPool2D

    tower_1 = Conv(filters=nfilters, kernel_size=1, padding='same', activation='relu', data_format=IMAGE_ORDERING)(img_input)

    tower_2 = Conv(filters=nfilters, kernel_size=3, padding='same', activation='relu', data_format=IMAGE_ORDERING)(tower_1)

    if not only_3x3_filters:
        tower_3 = Conv(filters=nfilters, kernel_size=5, padding='same', activation='relu', data_format=IMAGE_ORDERING)(tower_1)
    else:
        tower_3 = Conv(filters=nfilters, kernel_size=3, padding='same', activation='relu', data_format=IMAGE_ORDERING)(tower_2)

    tower_4 = MaxPool(pool_size=3, strides=1, padding='same', data_format=IMAGE_ORDERING)(img_input)
    tower_4 = Conv(filters=nfilters, kernel_size=1, padding='same', activation='relu', data_format=IMAGE_ORDERING)(tower_4)

    return Concatenate(axis=1)([tower_1, tower_2, tower_3, tower_4])


def basic_naive_inception(img_input, nfilters=64, IMAGE_ORDERING='channels_first', only_3x3_filters=False):
    Conv = Conv3D if len(img_input.shape) == 5 else Conv2D
    MaxPool = MaxPool3D if len(img_input.shape) == 5 else MaxPool2D

    tower_1 = Conv(filters=nfilters, kernel_size=1, padding='same', activation='relu', data_format=IMAGE_ORDERING)(img_input)

    tower_2 = Conv(filters=nfilters, kernel_size=3, padding='same', activation='relu', data_format=IMAGE_ORDERING)(img_input)

    if not only_3x3_filters:
        tower_3 = Conv(filters=nfilters, kernel_size=5, padding='same', activation='relu', data_format=IMAGE_ORDERING)(img_input)
    else:
        tower_3 = Conv(filters=nfilters, kernel_size=3, padding='same', activation='relu', data_format=IMAGE_ORDERING)(tower_2)

    tower_4 = MaxPool(pool_size=3, strides=1, padding='same', data_format=IMAGE_ORDERING)(img_input)

    return Concatenate([tower_1, tower_2, tower_3, tower_4])

def encode_inception(img_input, naive=False, IMAGE_ORDERING='channels_first', only_3x3_filters=False):
    fn = basic_naive_inception if naive else basic_rdim_inception

    MaxPool = MaxPool3D if len(img_input.shape) == 5 else MaxPool2D

    layer_1 = fn(img_input, 8, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)

    layer_2 = MaxPool(pool_size=3, strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_1)
    layer_2 = fn(layer_2, 16, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)

    layer_3 = MaxPool(pool_size=3, strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_2)
    layer_3 = fn(layer_3, 16, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)

    layer_4 = MaxPool(pool_size=3, strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_3)
    layer_4 = fn(layer_4, 32, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)

    layer_5 = MaxPool(pool_size=3, strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_4)
    layer_5 = fn(layer_5, 32, IMAGE_ORDERING=IMAGE_ORDERING, only_3x3_filters=only_3x3_filters)

    return [layer_1, layer_2, layer_3, layer_4, layer_5]


