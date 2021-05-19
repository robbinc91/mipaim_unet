import tensorflow as tf
from model import inception_unet
from utils import *
import pickle
import keras
from common import *



if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()

    #model_ = inception_unet()
    #model_ = inception_unet(shape=MNI_SHAPE, only_3x3_filters=ONLY_3X3_FILTERS, dropout=0.01)
    model_ = inception_unet(shape=REDUCED_MNI_SHAPE, only_3x3_filters=ONLY_3X3_FILTERS)
    #model_ = inception_unet(shape=(1, 192, 224))


    model_.compile(optimizer='adam',
                   loss=dice_loss,
                   metrics=[dice_coefficient])
    model_.summary()

    #exit(0)


    #from keras.utils.vis_utils import plot_model
    #plot_model(model_, to_file='small_mni_space_model_plot.png', show_shapes=True)


    #exit(0)

    t1_paths, seg_paths = hammers_2017_data_preprocessed_train_reduced(HAMMERS_ROOT)

    X = []
    y = []

    for t1_, seg_ in zip(t1_paths, seg_paths):
        #T1 = histeq(to_uint8(get_data_with_skull_scraping(t1_)))
        T1 = to_uint8(get_data(t1_))
        X.append(T1[None, ...])

        #y.append(np.array(get_data(seg_) == label).astype(np.uint8)[None, ...])
        y_ = to_uint8(get_data(seg_))
        y.append(y_[None, ...])

    X = np.array(X)
    y = np.array(y)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=10,
                                                           restore_best_weights=True,
                                                           baseline=0.07)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                             'weights/unet_3d_inception/all/model.{epoch:02d}.val_dice_coefficient={val_dice_coefficient:.5f}.h5',
                             monitor='val_dice_coefficient',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='max',
                             period=1
                         )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='./logs/all'
    )

    learning_rate_callback = keras.callbacks.LearningRateScheduler(step_decay)

    callbacks = [
        model_checkpoint_callback,
        early_stop_callback,
        tensorboard_callback,
        learning_rate_callback
    ]

    print('start fitting')

    history = model_.fit(x=X, y=y, validation_split=0.25, epochs=EPOCHS, batch_size=1, callbacks=callbacks)
    with open('history/unet_3d_inception_trainHistoryDict' + str(label) + '.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)



