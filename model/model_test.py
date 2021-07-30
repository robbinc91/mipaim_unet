from model import parcellation_inception_unet, parcellation_inception_unet_reduced, classification_model
from common import *

if __name__ == '__main__':
    _model = classification_model(only_3x3_filters=True, dropout=0.2)
    _model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    _model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(_model, to_file='classification_model.png', show_shapes=True)
