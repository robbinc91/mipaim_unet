import os
import numpy as np
import nibabel as nib
from pathlib import Path

def dice_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    
    return (2.0 * intersection) / union

def multilabel_to_binary(mask, target_labels=None):
    if target_labels is None:
        binary_mask = (mask > 0).astype(np.uint8)
    else:
        binary_mask = np.isin(mask, target_labels).astype(np.uint8)

    return binary_mask

def calculate_average_dsc(single_mask_path, multilabel_dir, target_labels=None):
    mask_img = nib.load(single_mask_path)
    binary_mask = mask_img.get_fdata().astype(np.uint8)

    binary_mask = (binary_mask > 0).astype(np.uint8)

    seg_dir = Path(multilabel_dir)
    seg_files = list(seg_dir.glob('*-parcellation-brainstem_mni_corrected.nii.gz'))

    if not seg_files:
        return -1
    
    dice_scores = {}

    for seg_file in seg_files:
        try:
            seg_img = nib.load(str(seg_file))
            multilabel_data = seg_img.get_fdata().astype(np.uint8)

            binary_pred = multilabel_to_binary(multilabel_data, target_labels)
            if binary_pred.shape != binary_mask.shape:
                continue

            score = dice_score(binary_mask, binary_pred)
            dice_scores[seg_file.name] = score

        except Exception as e:
            print(e)
            continue

    return dice_scores

if __name__ == '__main__':
    single_mask_path = 'D:\\university\\phd\\tests\\templates\\brainstem_mask.nii.gz'
    multilabel_directory = 'D:\\university\\phd\\tests\\test_images\\'

    scores = calculate_average_dsc(single_mask_path, multilabel_directory)

    if scores:
        all_scores = list(scores.values())
        average_dice = np.mean(all_scores)
        std_dice = np.std(all_scores)
        min_dice = np.min(all_scores)
        max_dice = np.max(all_scores)
        med_dice = np.median(all_scores)
        print(f"""Results:
All DSC: {all_scores}
Average DSC: {average_dice}
Min DSC: {min_dice}
Max DSC: {max_dice}
DSC stdev: {std_dice}
Median DSC: {med_dice}
""")
    else:
        print('no valid files encountered')
            

