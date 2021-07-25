#!/usr/bin/env bash

function help {
    echo -e "Usage: perform_preprocess.sh -i IMAGE -o OUTPUT_DIR [-n N4 -m BRAIN_MASK -x SEG_IMAGE]\n"

    echo -e "Preprocessing pipeline\n"
    echo -e "Performs (1) ROBEX (2) Weighted N4 (4) MNI registration. The brain"
    echo -e "mask calculated by ROBEX is only used for a better N4. The output"
    echo -e "image is NOT skull-stripped. Weighted N4 will apply N4 using the"
    echo -e "smoothed brain mask as weights (fit the bias field from the weighted"
    echo -e "pixel intensities).\n"

    echo -e "Args:"
    echo -e "    -i IMAGE: The input MPRAGE image"
    echo -e "    -o OUTPUT_DIR: The output directory. The inhomogeneity-corrected"
    echo -e "        image is {image_name}_n4.nii.gz and the MNI aligned image is"
    echo -e "        {image_name}_n4_mni.nii.gz"
    echo -e "    -n N4: If specified, the pipeline skips N4 and uses this image as"
    echo -e "         the input to the MNI registration."
    echo -e "    -m BRAIN_MASK: If specified, MNI will incorporate this in the"
    echo -e "         MNI registration (only use pixels in the mask for the loss)"
	echo -e "    -x SEG_IMAGE: If specified, labels for registering too"
}

while getopts ":hi:o:n:m:x:" opt; do
    case $opt in
        h)
            help
            exit 0
            ;;
        i)
            image=$OPTARG
            ;;
        o)
            output_dir=$OPTARG
            ;;
        n)
            n4_image=$OPTARG
            ;;
        m)
            brain_mask=$OPTARG
            ;;
		x)
			seg_image=$OPTARG
			;;
        \?)
            echo "Invalid option: $OPTARG"
            help
            exit 1
            ;;
    esac
done

echo N4 correction...
if [ -z $n4_image ]; then
    n4_dir=$output_dir/n4
    mkdir -p $n4_dir
    n4_image=$n4_dir/$(basename $image | sed "s/\.nii.*/_n4.nii.gz/")
    brain_mask=$(echo $n4_image | sed "s/_n4\.nii.*/_mask.nii.gz/")
    perform_n4.sh -i $image -o $n4_dir
fi

echo MNI registraion...
mni_dir=$output_dir/mni
mkdir -p $mni_dir
if [ -z $brain_mask ]; then
	if [ -z $seg_image ]; then
		perform_mni.sh -i $n4_image -o $mni_dir -t > /dev/null
	else
		perform_mni.sh -i $n4_image -o $mni_dir -x $seg_image -t > /dev/null
	fi
else
	if [ -z $seg_image ]; then
		perform_mni.sh -i $n4_image -o $mni_dir -t -m $brain_mask > /dev/null
	else
		perform_mni.sh -i $n4_image -o $mni_dir -x $seg_image -t -m $brain_mask > /dev/null
	fi
fi

