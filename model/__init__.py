from model.model import unet, \
    model_thresholding, \
    inception_unet, \
    parcellation_inception_unet, \
    parcellation_inception_unet_reduced, \
    classification_model, \
    parcellation_inception_unet_2, \
    inception_unet_semantic_segmentation, \
    mipaim_unet

from model.acapulco import build_acapulco_parcellation_unet
from model.swinunetr.swinunetr import SwinUNETR
from model.openmapt1.model import OpenMapT1UNet