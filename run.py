from utils.losses import *
from utils.preprocess import *
from keras.models import load_model


def inception_unet_predict(t1_path, label, output_path=None):
    model = load_model('weights/unet_3d_inception/labelinc/model_' + str(label) + ".h5",
                       custom_objects={'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss})
    #input = histeq(to_uint8(get_data_with_skull_scraping(t1_path)))[None, None, ...]
    input = to_uint8(get_data(t1_path))[None, None, ...]
    output = model.predict(input)

    if output_path:
        import nibabel as nib

        orig_ = nib.load(t1_path)
        affine = orig_.affine
        pred_ = nib.Nifti1Image(output, affine)
        nib.save(pred_, output_path)

    return output.squeeze()


def unet_predict(t1_path, label, output_path=None):
    model = load_model('weights/unet_3d/model_' + str(label) + ".h5",
                       custom_objects={'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss})
    #input = histeq(to_uint8(get_data_with_skull_scraping(t1_path)))[None, None, ...]
    input = to_uint8(get_data(t1_path))[None, None, ...]
    output = model.predict(input)

    if output_path:
        import nibabel as nib

        orig_ = nib.load(t1_path)
        affine = orig_.affine
        pred_ = nib.Nifti1Image(output, affine)
        nib.save(pred_, output_path)

    return output.squeeze()


def predict(T1_path, FLAIR_path, IR_path, label):
    model = load_model("weights/label" + str(label) + "/best.h5",
                       custom_objects={'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss})
    if label in [1, 3, 5]:
        T1 = get_data_with_skull_scraping(T1_path)
        if label == 5:
            X = np.array((T1 >= 10) & (T1 < 110)).astype(
                np.uint8)[None, None, ...]
        elif label == 3:
            X = np.array(T1 >= 150).astype(np.uint8)[None, None, ...]
        else:
            X = np.array((T1 >= 80) & (T1 < 160)).astype(
                np.uint8)[None, None, ...]
        y_pred = model.predict(X)
    else:
        T1 = histeq(to_uint8(get_data_with_skull_scraping(T1_path)))[
            None, None, ...]
        IR = IR_to_uint8(get_data(IR_path))[None, None, ...]
        FLAIR = to_uint8(get_data(FLAIR_path))[None, None, ...]
        y_pred = model.predict([T1, FLAIR, IR])
    return y_pred.squeeze()
