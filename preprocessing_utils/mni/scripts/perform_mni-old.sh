#!/usr/bin/env bash

if [ -z $MNIDIR ]; then
    MNIDIR=/opt/mni/template
fi

function usage() {
    echo "Register images to ICBM 2009c nonlinear symmetric template with ANTs."
    echo "A tool 'avscale' from FSL is also used to extract rigid tranfomration."
    echo "The the transformation between the first input image and the template"
    echo "and the transformations between the following images and the first input"
    echo "image are estimated. Image mask can be specified to increase accuracy"
    echo "when the brain images are cut-off partially."
    echo ""
    echo "Suppose the image is ${image}.nii.gz, then:"
    echo "    1. \${image}_Composite.h5: The transform to MNI space"
    echo "    2. \${image}_InverseComposite.h5: The transform from MNI to naive space"
    echo "    3. \${image}_mni.nii.gz: The transformed image in MNI space"
    echo ""
    echo "Usage: perform_mni.sh -i IMAGE1 [-i IMAGE2 ...] -o OUTDIR1 [-o OUTDIR2 ...]\\"
    echo "           [-m MASK1 [-m MASK2 ...]] -r -t -v"
    echo "Args:"
    echo "    -i IMAGE: The image to transform"
    echo "    -o OUTDIR: The output directory"
    echo "    -m MASK: The brain mask of IMAGE; if not specified, only "
    echo "        the mask of the MNI image is used in registraion"
    echo "    -r: Use random sampling"
    echo "    -t: Transform mask"
    echo "    -v: Verbose"
}

function add_suffix() {
    local string=$1
    local suffix=$2
    echo $(echo $string | sed "s/\.nii\(\.gz\)*$/${suffix}\.nii\.gz/")
}

function write_transform() {
    local trans=$1
    local output=$2
    local nums=($trans)
    local rot="${nums[@]:0:3} ${nums[@]:4:3} ${nums[@]:8:3}"
    local tr="${nums[3]} ${nums[7]} ${nums[11]}"
    echo "#Insight Transform File V1.0" > $output
    echo "#Transform 0" >> $output
    echo "Transform: AffineTransform_double_3_3" >> $output
    echo "Parameters: $rot $tr" >> $output
    echo "FixedParameters: 0 0 0" >> $output
}

function affine_init() {
    local moving_image=$1
    local fixed_image=$2
    local output_trans=$3
    
    local output_dir=$(dirname $output_trans)
    local resampled_moving=$output_dir/$(add_suffix $(basename $moving_image) _resampled)
    local resampled_fixed=$output_dir/$(add_suffix $(basename $fixed_image) _resampled)

    ResampleImageBySpacing 3 $moving_image $resampled_moving 4 4 4 1 > /dev/null
    ResampleImageBySpacing 3 $fixed_image $resampled_fixed 4 4 4 1 > /dev/null

    local init_affine=$output_dir/$(basename $output_trans | sed "s/\(\.txt\)$/_affine.mat/")
    local converted_affine=$output_dir/$(basename $init_affine | sed "s/\(\.mat\)$/_converted.mat/")

    antsAffineInitializer 3 $resampled_fixed $resampled_moving $init_affine 15 0.1 0 10 > /dev/null
    ConvertTransformFile 3 $init_affine $converted_affine --hm > /dev/null
    write_transform "$(avscale --allparams $converted_affine | sed -n "2,5p")" $output_trans
}

declare -a images
declare -a masks
declare -a output_dirs
transform_mask=false
sampling=Regular
verbose=false

while getopts ":hi:o:m:trv" opt; do
    case ${opt} in
    h)
        usage
        exit 0
        ;;
    i)
        images+=("$OPTARG")
        ;;
    o)
        output_dirs+=("$OPTARG")
        ;;
    m)
        masks+=("$OPTARG")
        ;;
    r)
        sampling=Random
        ;;
    t)
        transform_mask=true
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

if [ -z $images ] || [ -z $output_dirs ]; then
    echo -e "Not enough input arguments\n"
    usage
    exit 1
fi

image=${images[0]}
other_images=(${images[@]:1})
output_dir=${output_dirs[0]}
other_output_dirs=(${output_dirs[@]:1})
mask=${masks[0]}
other_masks=(${masks[@]:1})

prefix=$output_dir/$(basename ${image/.nii*/})
init_trans=${prefix}_init.txt
stage1_prefix=${prefix}_stage-1_
stage2_prefix=${prefix}_stage-2_
mni_image=$MNIDIR/mni_icbm152_2009c_t1_1mm.nii.gz
mni_regmask=$MNIDIR/mni_icbm152_2009c_t1_1mm_registration_mask.nii.gz
mni_brainmask=$MNIDIR/mni_icbm152_2009c_t1_1mm_brain_mask.nii.gz
output=${prefix}_mni.nii.gz

command="affine_init $image $mni_image $init_trans"
echo $command
eval $command

command="antsRegistration -d 3 -o $stage1_prefix -n Linear"
command="$command -r [$init_trans,0]"
command="$command -m MI[$mni_image,$image,1.0,32,$sampling,0.1]"
command="$command -t Rigid[0.1]"
command="$command -c [1000x500x250x125,1e-6,10]"
command="$command -s 3x2x1x0vox -f 8x4x2x1"
command="$command -w [0.01,0.99] --float 1 -a 1 -z 1 -i 0 -u -x $mni_regmask" 

echo $command
eval $command

command="antsRegistration -d 3 -o $stage2_prefix -n Linear"
command="$command -r [${stage1_prefix}Composite.h5,0]"
command="$command -m MI[$mni_image,$image,1.0,32,$sampling,0.1]"
command="$command -t Rigid[0.1]"
command="$command -c [1000x500x250x125,1e-6,10]"
command="$command -s 3x2x1x0vox -f 8x4x2x1"
command="$command -w [0.01,0.99] --float 1 -a 1 -z 1 -i 0 -u -x $mni_brainmask"

# if [ -z $mask ]; then
#     command="$command -x $mni_brainmask"
# else
#     command="$command -x [$mni_brainmask,$mask]"
# fi
echo $command
eval $command

command="antsApplyTransforms -d 3 -i $image -r $mni_image -o $output \
    -n Linear -t ${stage2_prefix}Composite.h5 --float"
echo $command
eval $command
if $transform_mask; then
    mask_prefix=$output_dir/$(basename ${mask/.nii*/})
    mask_output=${mask_prefix}_mni.nii.gz
    command="antsApplyTransforms -d 3 -i $mask -r $mni_image -o $mask_output \
        -n NearestNeighbor -t ${stage2_prefix}Composite.h5 --float"
    echo $command
    eval $command
fi

for i in ${!other_images[@]}; do
    oi=${other_images[$i]}
    om=${other_masks[$i]}
    od=${other_output_dirs[$i]}
    op=$od/$(basename ${oi/.nii*/})
    op_stage1=${op}_stage-1_
    op_stage2=${op}_stage-2_
    ot=${op}_init.txt
    oo=${op}_mni.nii.gz
    
    command="affine_init $oi $image $ot"
    echo $command
    eval $command

    command="antsRegistration -d 3 -o $op_stage1 -n Linear"
    command="$command -r [$ot,0]"
    command="$command -m MI[$image,$oi,1.0,32,$sampling,0.1]"
    command="$command -t Rigid[0.1]"
    command="$command -c [1000x500x250x125,1e-6,10]"
    command="$command -s 3x2x1x0vox -f 8x4x2x1"
    command="$command -w [0.01,0.99] --float 1 -a 1 -z 1 -i 0 -u" 

    echo $command
    eval $command

    command="antsRegistration -d 3 -o $op_stage2 -n Linear"
    command="$command -r [${op_stage1}Composite.h5,0]"
    command="$command -m MI[$image,$oi,1.0,32,$sampling,0.1]"
    command="$command -t Rigid[0.1]"
    command="$command -c [1000x500x250x125,1e-6,10]"
    command="$command -s 3x2x1x0vox -f 8x4x2x1"
    command="$command -w [0.01,0.99] --float 1 -a 1 -z 1 -i 0 -u"
    if [ ! -z "$mask" ]; then
        # if [ -z $om ]; then
        #     command="$command -x [$mask,$om]"
        # else
        command="$command -x $mask"
        # fi
    fi

    echo $command
    eval $command
    
    command="antsApplyTransforms -d 3 -i $oi -r $mni_image -o $oo \
        -n Linear -t ${stage2_prefix}Composite.h5 -t ${op_stage2}Composite.h5 \
        --float"
    echo $command
    eval $command

    if $transform_mask; then
        mask_prefix=$od/$(basename ${om/.nii*/})
        mask_output=${mask_prefix}_mni.nii.gz
        command="antsApplyTransforms -d 3 -i $om -r $mni_image -o $mask_output \
            -n NearestNeighbor -t ${stage2_prefix}Composite.h5 \
            -t ${op_stage2}Composite.h5 --float"
        echo $command
        eval $command
    fi
done
