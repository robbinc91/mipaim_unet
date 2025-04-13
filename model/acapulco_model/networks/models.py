# -*- coding: utf-8 -*-

from .layers import NormalizationFactory, ActivationFactory


class ConvBlockFactory:
    """Abstract class to create convolution models

    Attributes:
        kernel_size (int or (3,) tuple): The convolution kernel shape
        Normalization (keras.Layer class): Normalization Layer class
        Activation (keras.Layer class): Activation Layer class

    """
    def __init__(self, kernel_size=3, dropout_rate=0.,
                 kernel_initialization='glorot_normal',
                 normalization='instance', normalization_kwargs=dict(),
                 activation='leaky_relu', activation_kwargs=dict()):
        """Initialize
        
        Call NormalizationFactory to create a normalization Layer; call
        ActivationFactory to create a activation Layer.

        Args:
            normalization (str): The normalization method; see
                NormalizationFactory for more details
            activation (str): The activation method; see ActivationFactory for
                more details

        """
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.kernel_initialization = kernel_initialization
        self.Normalization = NormalizationFactory.create(normalization,
                                                         **normalization_kwargs)
        self.Activation = ActivationFactory.create(activation,
                                                   **activation_kwargs)

    def create(self, input_shape, num_out_features):
        """Create model

        Args:
            input_shape ((3,) tuple of int): The shape of the input data
            num_out_features (int): The number of features of the output

        Returns:
            model (keras.Model): The created model

        """
        raise NotImplementedError 
