# CERSEGSYS: CERebellum SEGmentation SYStem 

## Average time

~3m30s

## Dependencies

Python dependencies:

1. tensorflow==2.0.0
2. keras==2.3.1
3. pydicom==2.0.0
4. h5py==2.9.0
5. nibabel==3.1.1
6. deepbrain
7. cv2==4.3.0

Other dependencies:

8. ANTS
9. ROBEX

## Algorithm

The following processing steps are performed:

1. **N4 bias field correction**. [ROBEX](https://www.nitrc.org/projects/robex/) is used to estimate a brain mask. This mask is then smoothed to generate a brain weight image. N4 from [ANTs](http://stnava.github.io/ANTs/) is used to perform the bias field correction with the weight image calculated above.

2. **MNI registration**. The images are rigidly registered to the [ICBM 2009c](http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009) nonlinear symmetric template using the [ANTs](http://stnava.github.io/ANTs/) package.

3. **Cerebellum segmentation**. The cerebellum of an MNI-registered MPRAGE image is segmented using a U-net with Inception modules. Until now, no postprocessing is applied, though may be a longest connected componented could improve results.

4. **Transform back to the original space**. Since parcellation is performed in MNI space, we additionally transform the segmentation into the original image space.


## Future steps

The following steps will be included in next versions:

1. **Volume of each region**. The volume of segmented region will be calculated in mm^3.

2. **Result visualization**. A GUI application is beeing developed for visualization purposes. Check app.py.

## Docker image installation

`.tar.gz` file is provided. Use the following command;

```bash
gunzip cersegsys.tar.gz
docker load --input cersegsys.tar
```

### Usage

```bash
# segmentation
docker run -v "$PWD:$PWD" -w "$PWD" -t --user $(id -u):$(id -g) --rm cersegsys -i a01.nii.gz -o ./a01output -z

# segmentation using custom model
docker run -v "$PWD:$PWD" -w "$PWD" -t --user $(id -u):$(id -g) --rm cersegsys -i a01.nii.gz -o ./a01output -z -m ./model.h5

# apply only preprocessing steps
docker run -v "$PWD:$PWD" -w "$PWD" -t --user $(id -u):$(id -g) --rm cersegsys -i a01.nii.gz -o ./a01output

# apply only preprocess, registering a provided mask too
docker run -v "$PWD:$PWD" -w "$PWD" -t --user $(id -u):$(id -g) --rm cersegsys -i a01.nii.gz -o ./a01output -x ./a01-cerebellum.nii.gz
```