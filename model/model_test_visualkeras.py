from model import parcellation_inception_unet, \
    parcellation_inception_unet_reduced, \
    classification_model, \
    inception_unet_semantic_segmentation,\
    unet, \
    inception_unet

from keras.models import load_model
from common import *
import visualkeras
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 32)

if __name__ == '__main__':
    model_load = 'E:\\university\\phd\\tests\\niftifiles\\mine\\segmentation\\segm.h5'
    #_model = inception_unet(shape=REDUCED_MNI_SHAPE_MINE, only_3x3_filters=ONLY_3X3_FILTERS, dropout=0.2,
    #                        filters_dim=[8, 8, 16, 32, 32])
    _model = load_model(model_load, custom_objects={'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss})
    #_model.compile(optimizer='adam',
    #               loss=dice_loss,
    #               metrics=[dice_coefficient])
    _model.summary()
    visualkeras.layered_view(_model, to_file='output.png', legend=True, font=font)
    visualkeras.graph_view(_model, to_file='graph.png')#, legend=True, font=font)
    #from keras.utils.vis_utils import plot_model
    #plot_model(_model, to_file='parcellating_network_model.png', show_shapes=True)
