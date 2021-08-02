import tensorflow as tf
from model import classification_model
from utils import *
import pickle
import keras
from common import *
import json
import os

if __name__ == '__main__':


    output_folder = 'weights/unet_3d_inception/20210730-classification/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 400 epochs per label
    EPOCHS = 400

    _label = 'classification'
    __output_folder = '{0}{1}/'.format(output_folder, _label)
    if not os.path.exists(__output_folder):
        os.mkdir(__output_folder)

    model_ = classification_model(dropout=0.2)

    model_.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    model_.summary()

    partition, outputs = create_cersegsys_partitions(label=_label, use_augmentation=True)
    train_generator = DataGenerator(partition['train'], outputs, batch_size=4, root=CERSEGSYS_2_ROOT,
                                    shuffle=True, histogram_equalization=True,
                                    in_folder='preprocessed_parcellation_final_2',
                                    is_segmentation=False)
    val_generator = DataGenerator(partition['validation'], outputs, batch_size=4, root=CERSEGSYS_2_ROOT,
                                  shuffle=True, histogram_equalization=True,
                                  in_folder='preprocessed_parcellation_final_2',
                                  is_segmentation=False)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        __output_folder + 'model.epoch={epoch:03d}.val_accuracy={val_accuracy:.5f}.h5',
        monitor='val_accuracy',
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
    model_.save(__output_folder + 'model-{0}.h5'.format(_label))


