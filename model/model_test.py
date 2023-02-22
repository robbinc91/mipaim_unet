from model import parcellation_inception_unet, \
    parcellation_inception_unet_reduced, \
    classification_model, \
    inception_unet_semantic_segmentation,\
    unet, \
    inception_unet, \
    mipaim_unet

from keras.models import load_model
from common import *
import visualkeras
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 32)

if __name__ == '__main__':

    #model_load = 'E:\\university\\phd\\tests\\niftifiles\\mine\\segmentation\\segm.h5'
    # _model = inception_unet(shape=REDUCED_MNI_SHAPE_MINE, only_3x3_filters=ONLY_3X3_FILTERS, dropout=0.2,
    #                        filters_dim=[16, 32, 64, 128, 256], skip_connections_treatment_number=2)

    # _model = inception_unet_semantic_segmentation(shape=REDUCED_MNI_SHAPE_MINE,
    #                                              only_3x3_filters=ONLY_3X3_FILTERS,
    #                                              dropout=0.2,
    #                                              filters_dim=[32, 32, 64, 128, 128],
    #                                              num_labels=28,
    #                                              skip_connections_treatment_number=1)
    #_model = load_model(model_load, custom_objects={'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss})

    # _model = inception_unet(shape=REDUCED_MNI_SHAPE_CERSEGSYS_PARCELLATION,
    #                        only_3x3_filters=ONLY_3X3_FILTERS,
    #                        dropout=0.3,
    #                        filters_dim=[16, 16, 32, 64, 64],
    #                        instance_normalization=True)
    # _model.compile(optimizer='adam',
    #               loss=dice_loss,
    #               metrics=[dice_coefficient])
    # _model.summary()
    #from keras.utils.vis_utils import plot_model
    #plot_model(_model, to_file='lesions_.png', show_shapes=True)
    #visualkeras.layered_view(_model, to_file='lesions_2.png', legend=True, font=font)
    #visualkeras.graph_view(_model, to_file='lesions_graph.png')

    _model = mipaim_unet(shape=REDUCED_MNI_SHAPE_CERSEGSYS_PARCELLATION,
                         only_3x3_filters=ONLY_3X3_FILTERS,
                         dropout=0.3,
                         filters_dim=[16, 16, 32, 64, 64],
                         instance_normalization=True,
                         num_labels=4)

    _model.compile(optimizer='adam',
                   loss=soft_dice_loss,
                   metrics=[soft_dice_score])
    _model.summary()
    visualkeras.graph_view(_model, to_file='mipaim_unet.png')
