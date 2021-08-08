import tensorflow as tf
from model import parcellation_inception_unet_reduced
from utils import *
import pickle
import keras
from common import *
import json
import os

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    LABELS = json.load(open('labels.json'))['labels']
    _labels = list(LABELS.values())[1:]

    output_folder = 'weights/unet_3d_inception/20210804-parcellation-binary-2/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 400 epochs per label
    EPOCHS = 400

    _label = 'parcellation'
    __output_folder = '{0}{1}/'.format(output_folder, _label)
    if not os.path.exists(__output_folder):
        os.mkdir(__output_folder)

    model_ = parcellation_inception_unet_reduced(labels=len(_labels),
                                                 final_droput=0.22,
                                                 only_3x3_filters=True)
    model_.compile(optimizer='adam',
                   loss=dice_loss_multilabel,
                   metrics=[dice_coefficient_multilabel])
    model_.summary()

    partition, outputs = create_cersegsys_partitions(label=_label, use_augmentation=True)
    train_generator = DataGenerator(partition['train'], outputs, batch_size=4, root=CERSEGSYS_2_ROOT,
                                    shuffle=True, histogram_equalization=True,
                                    in_folder='preprocessed_parcellation_final_2',
                                    binary=True,
                                    labels=_labels)
    val_generator = DataGenerator(partition['validation'], outputs, batch_size=4, root=CERSEGSYS_2_ROOT,
                                  shuffle=True, histogram_equalization=True,
                                  in_folder='preprocessed_parcellation_final_2',
                                  binary=True,
                                  labels=_labels)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        __output_folder + 'model.epoch={epoch:03d}.val_dice_coefficient_multilabel={val_dice_coefficient_multilabel:.5f}.h5',
        monitor='val_dice_coefficient_multilabel',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        period=1
    )

    callbacks = [
        model_checkpoint_callback,
        # early_stop_callback,
        # tensorboard_callback,
        # learning_rate_callback
    ]

    history = model_.fit_generator(generator=train_generator, validation_data=val_generator, epochs=EPOCHS,
                                   use_multiprocessing=True,
                                   callbacks=callbacks)
    model_.save(__output_folder + 'model-{0}.h5'.format('fparc'))


