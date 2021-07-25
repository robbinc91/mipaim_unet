#!/usr/bin/env bash

function usage() {
    echo "N4 bias field correction for brain MR images. Do ROBEX on the original"
    echo "image (it has a rough bias field correction included so N4 before ROBEX"
    echo "is not required); then apply N4 with the weight image as the smoothed"
    echo "brain mask. Suppose the input image is image.nii.gz, then the N4 corrected"
    echo "is image_n4.nii.gz, the mask is image_mask.nii.gz, the weight image used"
    echo "in N4 is image_mask_weight.nii.gz, and the stripped is image_stripped.nii.gz"
    echo "Use ROBEX and N4BiasFieldCorrection from ANTs."
    echo ""
    echo "Usage: perform_n4.sh -i IMAGE -o OUTPUT_DIR"
    echo ""
    echo "Args:"
    echo "    -i IMAGE: The image to correct"
    echo "    -o OUTPUT_DIR: The output directory"
    echo "    -v: Print the executed commands and command verbose"
    echo "    -h: Show this message"
}

verbose=false

while getopts ":hi:o:v" opt; do
    case ${opt} in
    h)
        usage
        exit 0
        ;;
    i)
        input=$OPTARG
        ;;
    o)
        output_dir=$OPTARG
        ;;
    v)
        verbose=true
        ;;
    \?)
        echo "Invalid Option: -$OPTARG" 1>&2
        exit 1
        ;;
    :)
        echo "Invalid option: $OPTARG requires an argument" 1>&2
        ;;
    esac
done

if [ -z $input ] || [ -z $output_dir ]; then
    echo -e "Not enough input arguments\n"
    usage
    exit 1
fi

if ! [ -d $output_dir ]; then
    mkdir -p $output_dir
fi

mask=$output_dir/$(basename $input | sed "s/\.nii.*/_mask\.nii\.gz/g")
stripped=$output_dir/$(basename $input | sed "s/\.nii.*/_stripped\.nii\.gz/g")
n4=$output_dir/$(basename $input | sed "s/\.nii.*/_n4\.nii\.gz/g")
weight=$output_dir/$(basename $mask | sed "s/\.nii.*/_weight\.nii\.gz/g")

command="runROBEX.sh $input $stripped $mask"
if $verbose; then
    echo $command
    eval $command
else
    eval $command > /dev/null
fi

command="calc_weight_image.py -i $input -m $mask -o $weight"
if $verbose; then
    echo $command
fi
eval $command

sf=$(calc_shrinking_factor.py $input)
command="N4BiasFieldCorrection -d 3 -i $input -c [50x50x50x50,0.0005] -s $sf"
command="$command -o $n4 -w $weight -b [200]"
if $verbose; then
    command="$command -v"
    echo $command
fi
eval $command
