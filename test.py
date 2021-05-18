import tensorflow as tf
import keras
from keras.models import load_model
from utils.losses import *
import os

checkpoint_path = 'weights\\unet_3d_inception\\all\\model.epoch={epoch:03d}.val_dice_coefficient={val_dice_coefficient:.5f}.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(checkpoint_dir)
print(latest)

load_model(latest,
           custom_objects={'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss})

