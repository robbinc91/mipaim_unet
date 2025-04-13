# -*- coding: utf-8 -*-

from functools import partial
from keras.activations import softmax, sigmoid
from keras.layers import Activation, Layer

from ..configs import Configuration


class LayerFactory:
    
    @classmethod
    def create(cls):
        raise NotImplementedError


class NormalizationFactory(LayerFactory):
    
    @classmethod
    def create(cls, type, **kwargs):
        config = Configuration()
        if type == 'instance':
            from keras_contrib.layers import InstanceNormalization
            result = partial(InstanceNormalization, axis=config.channel_axis, dtype='float32',
                             **kwargs)
        elif type == 'batch':
            from keras.layers import BatchNormalization
            result = partial(BatchNormalization, axis=config.channel_axis, dtype='float32',
                             **kwargs)
        return result


class ActivationFactory(LayerFactory):

    @classmethod
    def create(cls, type, **kwargs):
        config = Configuration()
        if type == 'softmax':
            from keras.layers import Softmax
            result = partial(Softmax, axis=config.channel_axis, **kwargs)
        elif type == 'sigmoid':
            result = partial(Activation, 'sigmoid', **kwargs)
        elif type == 'relu':
            result = partial(Activation, 'relu', **kwargs)
        elif type == 'leaky_relu':
            from keras.layers import LeakyReLU
            result = partial(LeakyReLU, **kwargs)
        return result
