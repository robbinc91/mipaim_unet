import cv2 as cv
import nibabel as nib
import numpy as np
from deepbrain import Extractor
import SimpleITK as sitk
from math import pi
import nibabel.processing


def get_data(path):
    return nib.load(path).get_data()


def get_data_with_skull_scraping(path, PROB=0.5):
    arr = nib.load(path).get_data()
    ext = Extractor()
    prob = ext.run(arr)
    mask = prob > PROB
    arr = arr * mask
    return arr


def histeq(data):
    for slice_index in range(data.shape[2]):
        data[:, :, slice_index] = cv.equalizeHist(data[:, :, slice_index])
    return data


def to_uint8(data):
    data = data.astype(np.float)
    data[data < 0] = 0
    return ((data - data.min()) * 255.0 / data.max()).astype(np.uint8)


def IR_to_uint8(data):
    data = data.astype(np.float)
    data[data < 0] = 0
    return ((data - 800) * 255.0 / data.max()).astype(np.uint8)


def n4_bias_field_correction(input_image, outputImage=None,
                             shrinkFactor=None,
                             mask_image=None,
                             numberOfIterations=None,
                             number_of_fitting_levels=None):
    '''
    :param input_image: image to correct
    :param outputImage: image to save results
    :param shrinkFactor: factor
    :param mask_image: mask
    :param numberOfIterations: number of iterations
    :param number_of_fitting_levels: fitting levels
    :return: tuple (corrected image, calculated bias field, log bias field)
    '''
    inputImage = sitk.ReadImage(input_image, sitk.sitkFloat32)
    image = inputImage

    if mask_image is not None:
        maskImage = sitk.ReadImage(mask_image, sitk.sitkUint8)
    else:
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

    if shrinkFactor is not None:
        image = sitk.Shrink(inputImage,
                            [shrinkFactor] * inputImage.GetDimension())
        maskImage = sitk.Shrink(maskImage,
                                [shrinkFactor] * inputImage.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    numberFittingLevels = 4

    if number_of_fitting_levels is not None:
        numberFittingLevels = number_of_fitting_levels

    if numberOfIterations is not None:
        corrector.SetMaximumNumberOfIterations([numberOfIterations] * numberFittingLevels)

    corrected_image = corrector.Execute(image, maskImage)

    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    bias_field = inputImage / sitk.Exp(log_bias_field)

    if outputImage is not None:
        sitk.WriteImage(corrected_image, outputImage)

    return corrected_image, bias_field, log_bias_field


def exhautive_registration(fixedImageFilter, movingImageFile):
    '''
    :param fixedImageFilter: fixed image template
    :param movingImageFile: image to register
    :return: transform and transformed image
    '''
    fixed = sitk.ReadImage(fixedImageFilter, sitk.sitkFloat32)
    moving = sitk.ReadImage(movingImageFile, sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    sample_per_axis = 12
    if fixed.GetDimension() == 2:
        tx = sitk.Euler2DTransform()
        # Set the number of samples (radius) in each dimension, with a
        # default step size of 1.0
        R.SetOptimizerAsExhaustive([sample_per_axis // 2, 0, 0])
        # Utilize the scale to set the step size for each dimension
        R.SetOptimizerScales([2.0 * pi / sample_per_axis, 1.0, 1.0])
    elif fixed.GetDimension() == 3:
        tx = sitk.Euler3DTransform()
        R.SetOptimizerAsExhaustive([sample_per_axis // 2, sample_per_axis // 2,
                                    sample_per_axis // 4, 0, 0, 0])
        R.SetOptimizerScales(
            [2.0 * pi / sample_per_axis, 2.0 * pi / sample_per_axis,
             2.0 * pi / sample_per_axis, 1.0, 1.0, 1.0])

    # Initialize the transform with a translation and the center of
    # rotation from the moments of intensity.
    tx = sitk.CenteredTransformInitializer(fixed, moving, tx)

    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)

    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    transform = R.Execute(fixed, moving)

    # print("-------")
    # print(outTx)
    # print("Optimizer stop condition: {0}"
    #      .format(R.GetOptimizerStopConditionDescription()))
    # print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    # print(" Metric value: {0}".format(R.GetMetricValue()))

    # sitk.WriteTransform(outTx, sys.argv[3])
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(transform)

    resampled = resampler.Execute(moving)

    return transform, resampled

def applyTransform(fixed_path, img_path, transform_path):
    moving = sitk.ReadImage(img_path, sitk.sitkFloat32)
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    trf = sitk.ReadTransform(transform_path)


    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(trf)

    resampled = resampler.Execute(moving)

    return resampled


def conform_image(path, shape, pixel_sp):
    return nibabel.processing.conform(nib.load(path), shape, pixel_sp)


def crop3d(image, bbox):
    roi = np.asarray(image.dataobj[bbox[0], bbox[1], bbox[2]])
    return roi


def uncrop3d(image, source_shape, source_bbox):
    uncropped = np.zeros(source_shape, dtype=image.dtype)
    uncropped[tuple(source_bbox)] = image[:,:,:]
    return uncropped
