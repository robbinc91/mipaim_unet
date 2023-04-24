from email import generator
import tensorflow as tf
from model import mipaim_unet
from utils import *
import keras
from common import *
import json
import os
from keras_contrib.layers import InstanceNormalization
from keras.models import load_model
import tensorflow.keras
from pathlib import Path
#from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# input()
keras.backend.set_image_data_format('channels_first')

if __name__ == '__main__':

    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.enable_eager_execution()

    LABELS = json.load(open('labels_c7_bp_cp.json'))['labels']
    output_folder = 'weights/mipaim_unet/20230421_big/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 400 epochs per label
    EPOCHS = 300
    start_epoch = 0
    retrain = False

    _label = 'seg'

    __output_folder = '{0}{1}/'.format(output_folder, _label)
    if not os.path.exists(__output_folder):
        os.makedirs(__output_folder)
    
    paths = sorted(Path(__output_folder).iterdir(), key=os.path.getmtime)

    #if False:#os.path.exists(__output_folder+'model-{0}.h5'.format(_label)):
    #  print('reloading trained model')
    #  model_ = load_model(__output_folder+'model-{0}.h5'.format(_label),
    #                       custom_objects={'soft_dice_score': soft_dice_score, 'soft_dice_loss': soft_dice_loss, 'InstanceNormalization': InstanceNormalization})
    if len(paths) > 0:
        last_file = str(paths[-1]).split(os.sep)[-1]
        if last_file == 'model-{0}.h5'.format(_label):
           if not retrain:
              print('model train has already finished')
              import sys
              sys.exit(0)
           else:
              print('restart full')
              model_ = load_model(__output_folder + 'model-{0}.h5'.format(_label),
                           custom_objects={'soft_dice_score': soft_dice_score, 'soft_dice_loss': soft_dice_loss, 'InstanceNormalization': InstanceNormalization})
              start_epoch = EPOCHS
              EPOCHS *= 2
              
        else:
           # parse name
            start_epoch = int(last_file[12:15])
            model_ = load_model(__output_folder + last_file,
                           custom_objects={'soft_dice_score': soft_dice_score, 'soft_dice_loss': soft_dice_loss, 'InstanceNormalization': InstanceNormalization})
            last_score = last_file[last_file.find('val_dice_score='):-3][15:]
            print(f'restart training from epoch {start_epoch} with val_dice_score {last_score}')
    else:
      model_ = mipaim_unet(
          shape=CERSEGSYS_7_SHAPE,
          only_3x3_filters=ONLY_3X3_FILTERS,
          dropout=0.3,
          filters_dim=[16, 32, 64, 128, 256],
          instance_normalization=True,
          num_labels=11)
      model_.compile(optimizer='adam',
                    loss=soft_dice_loss,
                    metrics=[soft_dice_score])
    model_.summary()

    partition, outputs = create_cersegsys_partitions(
        label=_label, 
        use_augmentation=True, 
        second_lbl='', 
        use_originals=False # not using original images, as there is a big chance of having them among the augmented ones
    )
    train_generator = DataGenerator(partition['train'],
                                    outputs,
                                    batch_size=1,
                                    root=CERSEGSYS_7_ROOT,
                                    shuffle=True,
                                    histogram_equalization=False,
                                    in_folder='data',
                                    is_segmentation=True,
                                    binary=False)
    val_generator = DataGenerator(partition['validation'],
                                  outputs,
                                  batch_size=1,
                                  root=CERSEGSYS_7_ROOT,
                                  shuffle=True,
                                  histogram_equalization=False,
                                  in_folder='data',
                                  is_segmentation=True,
                                  binary=False)

    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        __output_folder +
        'model.epoch={epoch:03d}.val_dice_score={soft_dice_score:.5f}.h5',
        monitor='soft_dice_score',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='max',
        #save_freq=20
    )

    learning_rate_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='logs/c7_bp_cp_big'
    )

    callbacks = [
        model_checkpoint_callback,
        # early_stop_callback,
        tensorboard_callback,
        #learning_rate_callback
    ]

    print('start fitting on {0}'.format(_label))

    model_.fit(x=train_generator,
               y=None,
               batch_size=1,
               validation_data=val_generator,
               epochs=EPOCHS,
               use_multiprocessing=False,
               callbacks=callbacks,
               initial_epoch=start_epoch
               )
    #history = model_.fit_generator(generator=train_generator,
    #                               validation_data=val_generator,
    #                               epochs=EPOCHS,
    #                               use_multiprocessing=False,
    #                               callbacks=callbacks)
    model_.save(__output_folder+'model-{0}.h5'.format(_label))
