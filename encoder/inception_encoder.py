from keras.layers import Conv3D, MaxPool3D, Concatenate, MaxPool2D, Conv2D, Add, Average, Multiply
from keras_contrib.layers import InstanceNormalization
from attention import cbam_block


def basic_rdim_inception(img_input,
                         nfilters=64,
                         IMAGE_ORDERING='channels_first',
                         only_3x3_filters=False,
                         carry_input=False,
                         kernel_initializer=None,
                         instance_normalization=True):
    
    #print('basic rdim inception')
    
    normalization_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1

    Conv = Conv3D if len(img_input.shape) == 5 else Conv2D
    MaxPool = MaxPool3D if len(img_input.shape) == 5 else MaxPool2D

    tower_1 = Conv(filters=nfilters, kernel_size=1, padding='same',
                   activation='relu', data_format=IMAGE_ORDERING, kernel_initializer=kernel_initializer)(img_input)
    
    if instance_normalization is True:
        tower_1 = InstanceNormalization()(tower_1)

    tower_2 = Conv(filters=nfilters, kernel_size=3, padding='same',
                   activation='relu', data_format=IMAGE_ORDERING, kernel_initializer=kernel_initializer)(tower_1)
    
    if instance_normalization is True:
        tower_2 = InstanceNormalization()(tower_2)

    if not only_3x3_filters:
        tower_3 = Conv(filters=nfilters, kernel_size=5, padding='same',
                       activation='relu', data_format=IMAGE_ORDERING, kernel_initializer=kernel_initializer)(tower_1)
    else:
        tower_3 = Conv(filters=nfilters, kernel_size=3, padding='same',
                       activation='relu', data_format=IMAGE_ORDERING, kernel_initializer=kernel_initializer)(tower_2)

    if instance_normalization is True:
        tower_3 = InstanceNormalization()(tower_3)

    tower_4 = MaxPool(pool_size=3, strides=1, padding='same',
                      data_format=IMAGE_ORDERING)(img_input)
    
    if instance_normalization is True:
        tower_4 = InstanceNormalization()(tower_4)

    tower_4 = Conv(filters=nfilters, kernel_size=1, padding='same',
                   activation='relu', data_format=IMAGE_ORDERING, kernel_initializer=kernel_initializer)(tower_4)
    
    if instance_normalization is True:
        tower_4 = InstanceNormalization()(tower_4)
    

    if carry_input is True:
        return Concatenate(axis=1)([img_input, tower_1, tower_2, tower_3, tower_4])

    return Concatenate(axis=1)([tower_1, tower_2, tower_3, tower_4])


def basic_naive_inception(img_input, nfilters=64, IMAGE_ORDERING='channels_first', only_3x3_filters=False, carry_input=False, kernel_initializer=None):
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
                     skip_connections_method='inception',  # 'inception', 'conv', 'attention'
                     instance_normalization=False,
                     kernel_initializer=None,
                     use_input_mask=False,
                     use_output_mask=False):

    #if len(img_input.shape) <= 5:
    #    use_input_mask = False

    if use_input_mask is True:
        _sum = Add()(img_input)
        _average = Average()(img_input)
        _concat = Concatenate(axis=1)(img_input)
        _conv_1 = Conv3D(filters=8,
                   activation='relu',
                   strides=1,
                   kernel_size=3,
                   padding='same',
                   data_format=IMAGE_ORDERING)(img_input[0])
        _conv_1 = InstanceNormalization(dtype='float32')(_conv_1)
        _conv_2 = Conv3D(filters=8,
                   activation='relu',
                   strides=1,
                   kernel_size=3,
                   padding='same',
                   data_format=IMAGE_ORDERING)(img_input[1])
        _conv_2 = InstanceNormalization(dtype='float32')(_conv_2)
        _conv_3 = Conv3D(filters=8,
                   activation='relu',
                   strides=1,
                   kernel_size=3,
                   padding='same',
                   data_format=IMAGE_ORDERING)(_sum)
        _conv_3 = InstanceNormalization(dtype='float32')(_conv_3)
        _conv_4 = Conv3D(filters=8,
                   activation='relu',
                   strides=1,
                   kernel_size=3,
                   padding='same',
                   data_format=IMAGE_ORDERING)(_average)
        _conv_4 = InstanceNormalization(dtype='float32')(_conv_4)
        
        
        _conc = Concatenate(axis=1)([_sum, _average, _concat, _conv_1, _conv_2, _conv_3, _conv_4])

        _conc = Conv3D(filters=1,
                   activation='relu',
                   strides=1,
                   kernel_size=3,
                   padding='same',
                   data_format=IMAGE_ORDERING)(_conc)
#        print('superconv_output_shape:', _conc.output_shape)
        img_input = InstanceNormalization(dtype='float32')(_conc)



    if naive:
        fn = basic_naive_inception
    elif naive is None:
        fn = cbam_block
    else:
        fn = basic_rdim_inception
    MaxPool = MaxPool3D if len(img_input.shape) == 5 else MaxPool2D

    normalization_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1

    if filters_dim is None:
        filters_dim = [8, 16, 32, 64, 128]

    if instance_normalization is True:
        img_input = InstanceNormalization()(img_input)

    layer_1 = fn(img_input,
                 filters_dim[0],
                 IMAGE_ORDERING=IMAGE_ORDERING,
                 only_3x3_filters=only_3x3_filters,
                 carry_input=carry_input,
                 kernel_initializer=kernel_initializer)

    layer_2 = MaxPool(pool_size=3, strides=2, padding='same',
                      data_format=IMAGE_ORDERING)(layer_1)
    
    if instance_normalization is True:
        layer_2 = InstanceNormalization()(layer_2)

    layer_2 = fn(layer_2,
                 filters_dim[1],
                 IMAGE_ORDERING=IMAGE_ORDERING,
                 only_3x3_filters=only_3x3_filters,
                 carry_input=carry_input,
                 kernel_initializer=kernel_initializer)

    layer_3 = MaxPool(pool_size=3, strides=2, padding='same',
                      data_format=IMAGE_ORDERING)(layer_2)
    
    if instance_normalization is True:
        layer_3 = InstanceNormalization()(layer_3)

    layer_3 = fn(layer_3,
                 filters_dim[2],
                 IMAGE_ORDERING=IMAGE_ORDERING,
                 only_3x3_filters=only_3x3_filters,
                 carry_input=carry_input,
                 kernel_initializer=kernel_initializer)

    layer_4 = MaxPool(pool_size=3, strides=2, padding='same',
                      data_format=IMAGE_ORDERING)(layer_3)
    
    if instance_normalization is True:
        layer_4 = InstanceNormalization()(layer_4)

    layer_4 = fn(layer_4,
                 filters_dim[3],
                 IMAGE_ORDERING=IMAGE_ORDERING,
                 only_3x3_filters=only_3x3_filters,
                 carry_input=carry_input,
                 kernel_initializer=kernel_initializer)

    layer_5 = MaxPool(pool_size=3, strides=2, padding='same',
                      data_format=IMAGE_ORDERING)(layer_4)
    
    if instance_normalization is True:
        layer_5 = InstanceNormalization()(layer_5)

    layer_5 = fn(layer_5,
                 filters_dim[4],
                 IMAGE_ORDERING=IMAGE_ORDERING,
                 only_3x3_filters=only_3x3_filters,
                 carry_input=carry_input,
                 kernel_initializer=kernel_initializer)

    for i in range(skip_connections_treatment_number):

        if skip_connections_method == 'conv' or skip_connections_method == 'inception':
            layer_1 = fn(layer_1,
                         filters_dim[0],
                         IMAGE_ORDERING=IMAGE_ORDERING,
                         only_3x3_filters=only_3x3_filters,
                         carry_input=carry_input,
                         kernel_initializer=kernel_initializer)

            layer_2 = fn(layer_2,
                         filters_dim[1],
                         IMAGE_ORDERING=IMAGE_ORDERING,
                         only_3x3_filters=only_3x3_filters,
                         carry_input=carry_input,
                         kernel_initializer=kernel_initializer)

            layer_3 = fn(layer_3,
                         filters_dim[2],
                         IMAGE_ORDERING=IMAGE_ORDERING,
                         only_3x3_filters=only_3x3_filters,
                         carry_input=carry_input,
                         kernel_initializer=kernel_initializer)

            layer_4 = fn(layer_4,
                         filters_dim[3],
                         IMAGE_ORDERING=IMAGE_ORDERING,
                         only_3x3_filters=only_3x3_filters,
                         carry_input=carry_input,
                         kernel_initializer=kernel_initializer)

            layer_5 = fn(layer_5,
                         filters_dim[4],
                         IMAGE_ORDERING=IMAGE_ORDERING,
                         only_3x3_filters=only_3x3_filters,
                         carry_input=carry_input,
                         kernel_initializer=kernel_initializer)
        else:
            layer_1 = cbam_block(layer_1)
            layer_2 = cbam_block(layer_2)
            layer_3 = cbam_block(layer_3)
            layer_4 = cbam_block(layer_4)
            layer_5 = cbam_block(layer_5)

        if instance_normalization is True:
            layer_1 = InstanceNormalization()(layer_1)
            layer_2 = InstanceNormalization()(layer_2)
            layer_3 = InstanceNormalization()(layer_3)
            layer_4 = InstanceNormalization()(layer_4)
            layer_5 = InstanceNormalization()(layer_5)

    return [layer_1, layer_2, layer_3, layer_4, layer_5]
