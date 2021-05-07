import tensorflow as tf
from model import inception_unet
from utils import *
import pickle
import keras
from common import *



if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()

    #model_ = inception_unet()
    model_ = inception_unet(shape=(1, 192, 224, 192))


    model_.compile(optimizer='adam',
                   loss=dice_loss,
                   metrics=[dice_coefficient])
    model_.summary()

    # exit(0)

    from keras.utils.vis_utils import plot_model
    plot_model(model_, to_file='mni_space_model_plot.png', show_shapes=True)

    exit(0)

    #t1_paths, seg_paths = mrbrains2018_data_train(ROOT)
    t1_paths, seg_paths = hammers_2017_data_train(HAMMERS_ROOT)

    #t1_val, seg_val = mrbrains2018_data_val(ROOT)
    t1_val_paths, seg_val_paths = hammers_2017_data_val(HAMMERS_ROOT)

    X = []
    y = []

    t1_val = []
    seg_val = []

    for t1_, seg_ in zip(t1_paths, seg_paths):
        T1 = histeq(to_uint8(get_data_with_skull_scraping(t1_)))
        X.append(T1[None, ...])

        #y.append(np.array(get_data(seg_) == label).astype(np.uint8)[None, ...])
        y.append(np.array(get_data(seg_)).astype(np.uint8)[None, ...])

    X = np.array(X)
    y = np.array(y)

    for t1_val_, seg_val_ in zip(t1_val_paths, seg_val_paths):
        T1_val_ = histeq(to_uint8(get_data_with_skull_scraping(t1_val_)))[None, None, ...]
        t1_val.append(T1_val_)

        #seg_val.append(np.array(get_data(seg_val) == label).astype(np.uint8)[None, ...])
        seg_val.append(np.array(get_data(seg_val)).astype(np.uint8)[None, ...])

    t1_val = np.array(t1_val)
    seg_val = np.array(seg_val)

    #X_val = histeq(to_uint8(get_data_with_skull_scraping(t1_val)))[None, None, ...]
    #y_val = np.array(get_data(seg_val) == label).astype(np.uint8)[None, ...]

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #history = model_.fit(x=X, y=y, validation_data=(t1_val, seg_val), epochs=EPOCHS, batch_size=1,
    #                     callbacks=[keras.callbacks.ModelCheckpoint(
    #                         'weights/unet_3d_inception/label' + str(label) + '/Model.val_dice_coefficient={val_dice_coefficient:.5f}.h5',
    #                         monitor='val_dice_coefficient',
    #                         verbose=1,
    #                         save_best_only=True,
    #                         save_weights_only=False,
    #                         mode='max',
    #                         period=1
    #                     )])

    history = model_.fit(x=X, y=y, validation_data=(t1_val, seg_val), epochs=EPOCHS, batch_size=1,
                         callbacks=[keras.callbacks.ModelCheckpoint(
                             'weights/unet_3d_inception/all/Model.val_dice_coefficient={val_dice_coefficient:.5f}.h5',
                             monitor='val_dice_coefficient',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='max',
                             period=1
                         )])
    with open('history/unet_3d_inception_trainHistoryDict' + str(label) + '.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)



