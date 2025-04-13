# -*- coding: utf-8 -*-

import os
import json


class Singleton(type):
    """Singleton pattern
    
    There is only one instance of a singleton class

    """
    _instance = None
    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = super().__call__(*args, **kwargs)
        else:
            self._instance.__init__(*args, **kwargs)
        return self._instance


class Configuration(metaclass=Singleton):
    """Handle configuration

    Attributes:
        keras_config (dict): The configuration loaded from keras.json
        channel_axis (int): The axis of the channels
        data_format (str): "channels_first" or "channels_last"

    """
    def __init__(self):
        self.keras_config = {'image_data_format': 'channels_first'} #self._read_keras_config()
        self.channel_axis = 1 #self._get_channel_axis()
        self.data_format = 'channels_first' #self.keras_config['image_data_format']

    def _read_keras_config(self):
        keras_config_path = os.path.join(os.environ['HOME'],
                                         '.keras/keras.json')
        if os.path.isfile(keras_config_path):
            with open(keras_config_path) as config_file:
                keras_config = json.load(config_file)
        else:
            keras_config = {'image_data_format': 'channels_first'}

        return keras_config

    def _get_channel_axis(self):
        if self.keras_config['image_data_format'] == 'channels_first':
            return 1
        else:
            return -1
