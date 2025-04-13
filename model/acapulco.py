from glob import glob
import numpy as np
import os

import keras.backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam

from .acapulco_model.networks import InputFactory
from .acapulco_model.networks import AggregateOutputFactory
from .acapulco_model.networks import UNetDecoderFactory
from .acapulco_model.networks import UNetFactory
from .acapulco_model.networks import ResidueEncoderFactory
from .acapulco_model import Configuration



def build_acapulco_parcellation_unet(input_shape, num_classes):
    config = Configuration()

    activation_kwargs=dict(alpha=.1)
    input_factory = InputFactory(kernel_initialization='he_normal',
                                 normalization='instance',
                                 activation_kwargs=activation_kwargs)
    encoder_factory = ResidueEncoderFactory(kernel_initialization='he_normal',
                                            normalization='instance',
                                            activation_kwargs=activation_kwargs)
    decoder_factory = UNetDecoderFactory(kernel_initialization='he_normal',
                                         normalization='instance',
                                         dropout_rate=.2,
                                         activation_kwargs=activation_kwargs)
    output_factory = AggregateOutputFactory(kernel_initialization='he_normal',
                                            normalization='instance',
                                            activation='softmax')
    unet_factory = UNetFactory(input_factory, encoder_factory, decoder_factory,
                               output_factory, max_num_features=1024)
    unet = unet_factory.create(input_shape, 64,
                               num_classes, 4,
                               4)
    return unet
