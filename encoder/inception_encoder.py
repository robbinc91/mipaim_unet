from keras.layers import Conv3D, MaxPool3D, Concatenate, MaxPool2D, Conv2D
from keras_contrib.layers import InstanceNormalization
from attention import cbam_block

def basic_rdim_inception(img_input,
                         nfilters=64,
                         IMAGE_ORDERING='channels_first',
                         only_3x3_filters=False,
                         carry_input=False):

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

    if carry_input is True:
        return Concatenate(axis=1)([img_input, tower_1, tower_2, tower_3, tower_4])

    return Concatenate(axis=1)([tower_1, tower_2, tower_3, tower_4])


def basic_naive_inception(img_input, nfilters=64, IMAGE_ORDERING='channels_first', only_3x3_filters=False, carry_input=False):
    Conv = Conv3D if len(img_input.shape) == 5 else Conv2D
    MaxPool = MaxPool3D if len(img_input.shape) == 5 else MaxPool2D

    tower_1 = Conv(filters=nfilters,
                   kernel_size=1,
                   padding='same',
                   activation='relu',
                   data_format=IMAGE_ORDERING)(img_input)

    tower_2 = Conv(filters=nfilters,
                   kernel_size=3,
                   padding='same',
                   activation='relu',
                   data_format=IMAGE_ORDERING)(img_input)

    if not only_3x3_filters:
        tower_3 = Conv(filters=nfilters,
                       kernel_size=5,
                       padding='same',
                       activation='relu',
                       data_format=IMAGE_ORDERING)(img_input)
    else:
        tower_3 = Conv(filters=nfilters,
                       kernel_size=3,
                       padding='same',
                       activation='relu',
                       data_format=IMAGE_ORDERING)(tower_2)

    tower_4 = MaxPool(pool_size=3,
                      strides=1,
                      padding='same',
                      data_format=IMAGE_ORDERING)(img_input)

    if carry_input is True:
        return Concatenate(axis=1)([img_input, tower_1, tower_2, tower_3, tower_4])

    return Concatenate(axis=1)([tower_1, tower_2, tower_3, tower_4])


def encode_inception(img_input,
                     naive=False,
                     IMAGE_ORDERING='channels_first',
                     only_3x3_filters=False,
                     filters_dim=None,
                     carry_input=False,
                     skip_connections_treatment_number=0,
                     skip_connections_method='inception', # 'inception', 'conv', 'attention'
                     instance_normalization=False):

    fn = basic_naive_inception if naive else basic_rdim_inception
    MaxPool = MaxPool3D if len(img_input.shape) == 5 else MaxPool2D

    if filters_dim is None:
        filters_dim = [8, 16, 32, 64, 128]

    layer_1 = fn(img_input,
                 filters_dim[0],
                 IMAGE_ORDERING=IMAGE_ORDERING,
                 only_3x3_filters=only_3x3_filters,
                 carry_input=carry_input)

    normalization_axis = 1 if IMAGE_ORDERING is 'channels_first' else -1

    if instance_normalization is True:
        layer_1 = InstanceNormalization(axis=normalization_axis)(layer_1)

    layer_2 = MaxPool(pool_size=3, strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_1)
    layer_2 = fn(layer_2,
                 filters_dim[1],
                 IMAGE_ORDERING=IMAGE_ORDERING,
                 only_3x3_filters=only_3x3_filters,
                 carry_input=carry_input)

    if instance_normalization is True:
        layer_2 = InstanceNormalization(axis=normalization_axis)(layer_2)

    layer_3 = MaxPool(pool_size=3, strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_2)
    layer_3 = fn(layer_3,
                 filters_dim[2],
                 IMAGE_ORDERING=IMAGE_ORDERING,
                 only_3x3_filters=only_3x3_filters,
                 carry_input=carry_input)

    if instance_normalization is True:
        layer_3 = InstanceNormalization(axis=normalization_axis)(layer_3)

    layer_4 = MaxPool(pool_size=3, strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_3)
    layer_4 = fn(layer_4,
                 filters_dim[3],
                 IMAGE_ORDERING=IMAGE_ORDERING,
                 only_3x3_filters=only_3x3_filters,
                 carry_input=carry_input)

    if instance_normalization is True:
        layer_4 = InstanceNormalization(axis=normalization_axis)(layer_4)

    layer_5 = MaxPool(pool_size=3, strides=2, padding='same', data_format=IMAGE_ORDERING)(layer_4)
    layer_5 = fn(layer_5,
                 filters_dim[4],
                 IMAGE_ORDERING=IMAGE_ORDERING,
                 only_3x3_filters=only_3x3_filters,
                 carry_input=carry_input)

    if instance_normalization is True:
        layer_5 = InstanceNormalization(axis=normalization_axis)(layer_5)

    for i in range(skip_connections_treatment_number):

        if skip_connections_method is 'conv' or skip_connections_method is 'inception':
            layer_1 = fn(layer_1,
                        filters_dim[0],
                        IMAGE_ORDERING=IMAGE_ORDERING,
                        only_3x3_filters=only_3x3_filters,
                        carry_input=carry_input)

            layer_2 = fn(layer_2,
                        filters_dim[1],
                        IMAGE_ORDERING=IMAGE_ORDERING,
                        only_3x3_filters=only_3x3_filters,
                        carry_input=carry_input)

            layer_3 = fn(layer_3,
                        filters_dim[2],
                        IMAGE_ORDERING=IMAGE_ORDERING,
                        only_3x3_filters=only_3x3_filters,
                        carry_input=carry_input)

            layer_4 = fn(layer_4,
                        filters_dim[3],
                        IMAGE_ORDERING=IMAGE_ORDERING,
                        only_3x3_filters=only_3x3_filters,
                        carry_input=carry_input)

            layer_5 = fn(layer_5,
                        filters_dim[4],
                        IMAGE_ORDERING=IMAGE_ORDERING,
                        only_3x3_filters=only_3x3_filters,
                        carry_input=carry_input)
        else:
            layer_1 = cbam_block(layer_1)
            layer_2 = cbam_block(layer_2)
            layer_3 = cbam_block(layer_3)
            layer_4 = cbam_block(layer_4)
            layer_5 = cbam_block(layer_5)


    return [layer_1, layer_2, layer_3, layer_4, layer_5]


