#!/usr/bin/env bash

if [ -z $ALPHA ]; then
    ALPHA=1
fi
if [ -z $NUM_SLICES ]; then
    NUM_SLICES=10
fi

function usage() {
    echo "Image preprocessing for Deep Learning."
    echo ""
    echo "Usage: preprocess.sh -i IMAGE -m MODEL -o OUTPUT_DIR"
    echo "Args:"
    echo "    -i IMAGE: The input MPRAGE image"
    echo "    -o OUTPUT_DIR: The output directory. The MNI aligned image is "
    echo "        OUTPUT_DIR/mni/{image}_n4_mni.nii.gz."
	  echo "    -x SEG_IMAGE: Segmented image"
	  echo "    -z: Perform cerebellum segmentation "
	  echo "    -v: Calculate volumes"
  

}

model="default"
perf_segm=false
calc_volume=false

while getopts ":hi:o:m:x:zv:" opt; do
    case ${opt} in
    h)
        usage
        exit 0
        ;;
    i)
        image=$OPTARG
        ;;
    o)
        output_dir=$OPTARG
        ;;
    m)
        model=$OPTARG
        ;;
	  x)
		  seg_image=$OPTARG
		  ;;
		z)
		  perf_segm=true
		  ;;
		v)
		  calc_volume=true
		  ;;
    \?)
        echo "Invalid Option: -$OPTARG." 1>&2
		usage
        exit 1
        ;;
    :)
        echo "Invalid option: $OPTARG requires an argument." 1>&2
        ;;
    esac
done

if [ -z $seg_image ]; then
	perform_preprocess.sh -i $image -o $output_dir
else
	perform_preprocess.sh -i $image -o $output_dir -x $seg_image
fi

if $perf_segm; then
  echo "perform cerebellum segmentation"
  mni_dir=$output_dir/mni
  processed_dir=$output_dir/processed
  mni_image=$mni_dir/$(basename $image | sed "s/\.nii.*/_n4_mni.nii.gz/")
  mkdir -p $processed_dir

  # perform crop
  cropped_image=$processed_dir/$(basename $image | sed "s/\.nii.*/_n4_mni_crop.nii.gz/")
  command="python /usr/local/bin/utils/crop_file.py -i $mni_image -o $cropped_image > /dev/null"
  eval $command

  # perform segmentation
  segmented_image=$processed_dir/$(basename $image | sed "s/\.nii.*/_n4_mni_crop_segm.nii.gz/")
  command="python /usr/local/bin/predict.py -i $cropped_image -o $segmented_image -m $model > /dev/null"
  eval $command

  # perform uncropping
  uncropped_image=$processed_dir/$(basename $image | sed "s/\.nii.*/_n4_mni_crop_segm_uncrop.nii.gz/")
  command="python /usr/local/bin/utils/uncrop_file.py -i $segmented_image -o $uncropped_image > /dev/null"
  eval $command

  # unregister
  final_image=$output_dir/$(basename $image | sed "s/\.nii.*/_final_segm.nii.gz/")
  transform=$(echo $mni_image | sed "s/\.gz//" | sed "s/_mni\.nii/_stage-2_InverseComposite.h5/")
  #antsApplyTransforms -d 3 -i $uncropped_image -r $image -o $final_image -t [$transform,1] --float -n MultiLabel[1.0,3.0]
  antsApplyTransforms -d 3 -i $uncropped_image -r $image -o $final_image -n NearestNeighbor -t $transform --float


  #volume calculation
  echo "Volume calculation comes soon, as well as postprocessing. All done!"
fi