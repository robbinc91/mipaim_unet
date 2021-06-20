import nibabel

from utils import *
from common import *
import SimpleITK as sitk
import os



def do_preprocess(root):
    for i in range(32, 43):
        _name = 'a{:02d}'.format(i)
        print('On {0}'.format(_name))

        print('processing: ', _name)

        # Bias field Correction -----------------
        print(' Bias Field Correction...')
        corrected_image, bias_field, log_bias_field = n4_bias_field_correction(root + 'images/' + _name + '.nii.gz')
        sitk.WriteImage(corrected_image, root + 'pre/' + _name + '-bfc.nii.gz')
        sitk.WriteImage(bias_field, root + 'pre/' + _name + '-bf.nii.gz')
        sitk.WriteImage(log_bias_field, root + 'pre/' + _name + '-lbf.nii.gz')

        # Image Registration -------------------
        print(' Registration to MNI152')
        transform, resampled = exhautive_registration(IMG_ROOT + 'registration/MNI152_T1_1mm.nii.gz', root + 'pre/' + _name + '-bfc.nii.gz')
        sitk.WriteTransform(transform, root + 'pre/' + _name + '-transform.tfm')
        sitk.WriteImage(resampled, root + 'pre/' + _name + '-reg.nii.gz')

        # labels resampling ----------------------
        print(' Label Resampling')
        resampled_labels = applyTransform(IMG_ROOT + 'registration/MNI152_T1_1mm.nii.gz',
                                          root + 'images/' + _name + '-seg.nii.gz',
                                          root + 'pre/' + _name + '-transform.tfm')
        sitk.WriteImage(resampled_labels, root + 'pre/' + _name + '-reg-seg.nii.gz')

        # Image and labels resample ----------------------
        print(' Resample for a convenient shape')
        resized_img = conform_image(root + 'pre/' + _name + '-reg.nii.gz', (192, 224, 192), (1., 1., 1.))
        nib.save(resized_img, root + 'pre/' + _name + '-reg-res.nii.gz')

        resized_labels = conform_image(root + 'pre/' + _name + '-reg-seg.nii.gz', (192, 224, 192), (1., 1., 1.))
        nib.save(resized_labels, root + 'pre/' + _name + '-reg-seg-res.nii.gz')

        # Skull Stripping and histogram equalization -----------------------
        print(' Skull stripping and histogram equalization')
        no_skull = get_data_with_skull_scraping(root + 'pre/' + _name + '-reg-res.nii.gz')
        hist_equ = histeq(to_uint8(no_skull))
        affine = nibabel.load(root + 'pre/' + _name + '-reg-res.nii.gz').affine
        no_skull_img = nibabel.Nifti1Image(no_skull, affine)
        hist_eq_img = nibabel.Nifti1Image(hist_equ, affine)

        nibabel.save(no_skull_img, root + 'pre/' + _name + '-reg-res-no-skull.nii.gz')
        nibabel.save(hist_eq_img, root + 'pre/' + _name + '-reg-res-no-skull-hist-eq.nii.gz')

        img = nibabel.load(root + 'pre/' + _name + '-reg-res-no-skull-hist-eq.nii.gz')
        nibabel.save(img, root + 'final/' + _name + '-pre.nii.gz')
        seg = nibabel.load(root + 'pre/' + _name + '-reg-seg-res.nii.gz')
        nibabel.save(seg, root + 'final/' + _name + '-labels.nii.gz')

        # Mask with only cerebellum
        print(' Mask with cerebellum')
        seg = nibabel.load(root + 'pre/' + _name + '-reg-seg-res.nii.gz')
        cerebl = np.array(seg.get_data() == 17).astype(np.uint8) + np.array(seg.get_data() == 18).astype(np.uint8)
        cerebl = nibabel.Nifti1Image(cerebl, seg.affine)
        nibabel.save(cerebl, root + 'final/' + _name + '-cerebellum.nii.gz')

        # Mask with only brainstem
        #bst = np.array(seg.get_data() == 19).astype(np.uint8)
        #bst = nibabel.Nifti1Image(bst, seg.affine)
        #nibabel.save(bst, root + 'final/' + _name + '-brainstem.nii.gz')

        # Reduce image shape
        print(' Reduce dimensions')
        img = nib.load(root + 'final/' + _name + '-cerebellum.nii.gz')
        roi = np.asarray(img.dataobj[x_idx_range, y_idx_range, z_idx_range])
        cropped_img = nib.Nifti1Image(roi, affine=img.affine)
        nib.save(cropped_img, root + 'reduced/' + _name + '-cerebellum.nii.gz')

        img = nib.load(root + 'final/' + _name + '-pre.nii.gz')
        roi = np.asarray(img.dataobj[x_idx_range, y_idx_range, z_idx_range])
        cropped_img = nib.Nifti1Image(roi, affine=img.affine)
        nib.save(cropped_img, root + 'reduced/' + _name + '-pre.nii.gz')


def preprocess_single_image_process(folder, name, IMG_ROOT='E:\\university\\phd\\phd\\datasets\\'):
    path = folder + name

    # BFC
    corrected_image, bias_field, log_bias_field = n4_bias_field_correction(path)
    sitk.WriteImage(corrected_image, folder + '01 - bfc.nii.gz')
    sitk.WriteImage(log_bias_field, folder + '01 - lbf.nii.gz')


    # Registration
    transform, resampled = exhautive_registration(IMG_ROOT + 'registration/MNI152_T1_1mm.nii.gz',
                                                  folder + '01 - bfc.nii.gz')
    sitk.WriteTransform(transform, folder + '02 - transform.tfm')
    sitk.WriteImage(resampled, folder + '02 - reg.nii.gz')

    # Resampling
    resized_img = conform_image(folder + '02 - reg.nii.gz', (192, 224, 192), (1., 1., 1.))
    nib.save(resized_img, folder + '03 - reg-res.nii.gz')

    # Skull stripping
    affine = nibabel.load(folder + '03 - reg-res.nii.gz').affine

    no_skull = get_data_with_skull_scraping(folder + '03 - reg-res.nii.gz')
    no_skull_img = nibabel.Nifti1Image(no_skull, affine)
    nibabel.save(no_skull_img, folder + '04 - reg-res-no-skull.nii.gz')

    # Histogram equalization
    hist_equ = histeq(to_uint8(no_skull))
    hist_eq_img = nibabel.Nifti1Image(hist_equ, affine)
    nibabel.save(hist_eq_img, folder + '05 - reg-res-no-skull-hist-eq.nii.gz')

    img = nib.load(folder + '05 - reg-res-no-skull-hist-eq.nii.gz')
    roi = np.asarray(img.dataobj[x_idx_range, y_idx_range, z_idx_range])
    cropped_img = nib.Nifti1Image(roi, affine=img.affine)
    nib.save(cropped_img, folder + '06 - reg-res-no-skull-hist-eq-cropped.nii.gz')





def preprocess_single_image(img_path):
    tmp = img_path.split(os.sep)
    folder = os.sep.join(tmp[:-1]) + os.sep
    name = tmp[-1]
    preprocess_single_image_process(folder, name)



if __name__ == '__main__':
    do_preprocess(HAMMERS_ROOT)
    #preprocess_single_image('E:\\university\\phd\\phd\\datasets\\hammers_full_2017\\tests\\extra\\a31.nii.gz')
    pass
