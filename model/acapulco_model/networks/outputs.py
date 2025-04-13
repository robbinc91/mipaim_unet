# -*- coding: utf-8 -*-

from keras.layers import Input, Conv3D, UpSampling3D, Add, Flatten, Dense
from keras.layers import GlobalAveragePooling3D
from keras.models import Model

from .models import ConvBlockFactory
from ..configs import Configuration


class OutputFactory(ConvBlockFactory):

    def create(input_shapes, num_classes):
        """Abstract method to create a output block

        Args:
            input_shapes (list of (3,) shapes or a shape): The shapes of the
                input tensors. If not a list (shape is always a (3,) tuple!),
                there is only one input to this output block. If a list, the
                last item is the shape of the output tensor as well
            num_classes (int): The number of segmentation classes

        Returns:
            model (keras.Model): The created output block

        """
        raise NotImplementedError

    def _parse_input_shapes(self, input_shapes):
        if type(input_shapes) is not list:
            input_shapes = [input_shapes]
        return input_shapes


class SimpleOutputFactory(OutputFactory):
    """Create a simple output block
    
    input -> conv -> activ (softmax, etc.)

    """
    def create(self, input_shapes, num_classes):
        input_shapes = self._parse_input_shapes(input_shapes)
        input = Input(input_shapes[-1])
        output = Conv3D(num_classes, 1, padding='same', use_bias=False,
                        kernel_initializer=self.kernel_initialization)(output)
        output = self.Activation(dtype='float32')(output) 
        model = Model(inputs=input, outputs=output)
        return model


class AggregateOutputFactory(OutputFactory):
    """Create an aggregate output block
    
    Combine outputs of all decoders

    """
    def create(self, input_shapes, num_classes):
        input_shapes = self._parse_input_shapes(input_shapes)
        inputs = list()
        for i, input_shape in enumerate(input_shapes):
            input = Input(input_shape)
            inputs.append(input)
            conv = Conv3D(num_classes, 1, padding='same', use_bias=False,
                          kernel_initializer=self.kernel_initialization)(input)
            if i > 0:
                output = Add()([conv, output])
            else:
                output = conv
            if i < len(input_shapes) - 1:
                output = UpSampling3D(size=(2, 2, 2))(output)
        output = self.Activation(dtype='float32')(output) 
        model = Model(inputs=inputs, outputs=output)
        return model


class RegressionOutputFactory(ConvBlockFactory):

    def create(self, input_shape, num_values):
        input = Input(shape=input_shape)
        output = Flatten()(input)
        output = Dense(num_values, use_bias=False)(output)
        output = self.Activation(dtype='float32')(output)
        model = Model(inputs=input, outputs=output)
        return model


class RegressAvgOutputFactory(ConvBlockFactory):
    def create(self, input_shape, num_values):
        config = Configuration()
        input = Input(shape=input_shape)
        output = GlobalAveragePooling3D(data_format=config.data_format)(input)
        output = Dense(num_values, use_bias=False)(output)
        # output = self.Activation()(output)
        model = Model(inputs=input, outputs=output)
        return model
