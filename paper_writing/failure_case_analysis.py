# 2 Failure Case Analysis - Methodology
#This is a qualitative process, not a single code function.
#  Identify Candidates: Sort your test set results by Dice score (lowest first) or HD95 (highest first).
#  Visualize the Worst Performers: For the top 3-5 worst cases, generate comprehensive visualizations. Use a function like the one below to create a detailed plot for each failure case.
#
#
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from utils.utils import cersegsys_test_prt, cersegsys_train_prt

images_folder = 'D:\\university\\phd\\phd\\datasets\\cersegsys_7\\data\\'
segmentations_folder = 'D:\\university\\phd\\phd\\evals\\segmentations\\'
output_folder = 'D:\\university\\phd\\phd\\evals\\segmentation_visualizations\\'
model_names = ['mipaim', 'no_mipaim', 'acapulco', 'unet', 'openmapt1']

def load_case(image_path, label_path):
    """Load a single case from NIfTI files"""
    img = nib.load(image_path).get_fdata().astype(np.float32)
    lbl = nib.load(label_path).get_fdata().astype(np.int8)
    
    # Normalize image (example using 0-1 normalization)
    #img = (img - img.min()) / (img.max() - img.min())
    
    return img, lbl

colors = ['b', 'g', 'm', 'y', 'c']

def visualize_failure_all(original_image, gt_mask, pred_masks, slice_idx=None):

    if slice_idx is None:
        # Find the central slice with the largest area of the structure
        slice_idx = np.argmax(np.sum(gt_mask, axis=(1, 2))) 

    fig, axes = plt.subplots(1, 3, figsize=(10, 15))
    axes = axes.ravel()

    error_map_sagittal = np.zeros((*original_image.shape[1:3], 3)) # Create RGB image
    error_map_coronal = np.zeros((*original_image.shape[1:3], 3)) # Create RGB image
    error_map_axial = np.zeros((*original_image.shape[0:2], 3)) # Create RGB image


    axes[0].imshow(original_image[slice_idx], cmap='gray')
    axes[0].contour(gt_mask[slice_idx], colors='r', linewidths=1)
    axes[0].axis('off')
    axes[1].imshow(original_image[:, slice_idx, :], cmap='gray')
    axes[1].contour(gt_mask[:, slice_idx, :], colors='r', linewidths=1)
    axes[1].axis('off')
    axes[2].imshow(original_image[:, :, slice_idx], cmap='gray')
    axes[2].contour(gt_mask[:, :, slice_idx], colors='r', linewidths=1)
    axes[2].axis('off')


    for i, (model_name, pred_mask) in enumerate(pred_masks.items()):
        axes[0].contour(pred_mask[slice_idx], colors=colors[i], linewidths=1)
        axes[1].contour(pred_mask[:, slice_idx, :], colors=colors[i], linewidths=1)
        axes[2].contour(pred_mask[:, :, slice_idx], colors=colors[i], linewidths=1)

    plt.tight_layout()


def visualize_failure_case(original_image, gt_mask, pred_mask, slice_idx=None):
    """
    Creates a plot to visually analyze a segmentation failure.
    """
    if slice_idx is None:
        # Find the central slice with the largest area of the structure
        slice_idx = np.argmax(np.sum(gt_mask, axis=(1, 2))) 
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    # Original image
    error_map_sagittal = np.zeros((*original_image.shape[1:3], 3)) # Create RGB image
    error_map_coronal = np.zeros((*original_image.shape[1:3], 3)) # Create RGB image
    error_map_axial = np.zeros((*original_image.shape[0:2], 3)) # Create RGB image

    overlay_alpha = 1 - math.sqrt(0.5)


    axes[0].imshow(original_image[slice_idx], cmap='gray')
    axes[0].imshow(error_map_sagittal, alpha=0.5)
    axes[0].set_title('Sagittal')
    axes[0].axis('off')

    axes[1].imshow(original_image[:, slice_idx, :], cmap='gray')
    axes[1].imshow(error_map_coronal, alpha=0.5) # Overlay error map
    axes[1].set_title('Coronal')
    axes[1].axis('off')

    axes[2].imshow(original_image[:, :, slice_idx], cmap='gray')
    axes[2].imshow(error_map_axial, alpha=0.5) # Overlay error map
    axes[2].set_title('Axial')
    axes[2].axis('off')

    
    # Overlay (to see differences)
    axes[3].imshow(original_image[slice_idx], cmap='gray')
    axes[3].contour(gt_mask[slice_idx], colors='r', linewidths=2)
    axes[3].contour(pred_mask[slice_idx], colors='b', linewidths=2)
    axes[3].set_title('Overlay (GT Red, Pred Blue)')
    axes[3].axis('off')
    
    # Error Map (FP: Green, FN: Red)
     
    error_map_sagittal[pred_mask[slice_idx] > gt_mask[slice_idx]] = [0, 1, 0] # FP = Green
    error_map_sagittal[gt_mask[slice_idx] > pred_mask[slice_idx]] = [1, 0, 0] # FN = Red
    
    axes[6].imshow(original_image[slice_idx], cmap='gray')
    axes[6].imshow(error_map_sagittal, alpha=0.5) # Overlay error map
    axes[6].set_title('Errors: FP (Green), FN (Red)')
    axes[6].axis('off')


    # Overlay (to see differences)
    axes[4].imshow(original_image[:, slice_idx, :], cmap='gray')
    axes[4].contour(gt_mask[:, slice_idx, :], colors='r', linewidths=2)
    axes[4].contour(pred_mask[:, slice_idx, :], colors='b', linewidths=2)
    axes[4].set_title('Overlay (GT Red, Pred Blue)')
    axes[4].axis('off')
    
    # Error Map (FP: Green, FN: Red)
    
    error_map_coronal[pred_mask[:, slice_idx, :] > gt_mask[:, slice_idx, :]] = [0, 1, 0] # FP = Green
    error_map_coronal[gt_mask[:, slice_idx, :] > pred_mask[:, slice_idx, :]] = [1, 0, 0] # FN = Red
    
    axes[7].imshow(original_image[:, slice_idx, :], cmap='gray')
    axes[7].imshow(error_map_coronal, alpha=0.5) # Overlay error map
    axes[7].set_title('Errors: FP (Green), FN (Red)')
    axes[7].axis('off')

    # Overlay (to see differences)
    axes[5].imshow(original_image[:, :, slice_idx], cmap='gray')
    axes[5].contour(gt_mask[:, :, slice_idx], colors='r', linewidths=2)
    axes[5].contour(pred_mask[:, :, slice_idx], colors='b', linewidths=2)
    axes[5].set_title('Overlay (GT Red, Pred Blue)')
    axes[5].axis('off')
    
    # Error Map (FP: Green, FN: Red)
    
    error_map_axial[pred_mask[:, :, slice_idx] > gt_mask[:, :, slice_idx]] = [0, 1, 0] # FP = Green
    error_map_axial[gt_mask[:, :, slice_idx] > pred_mask[:, :, slice_idx]] = [1, 0, 0] # FN = Red
    
    axes[8].imshow(original_image[:, :, slice_idx], cmap='gray')
    axes[8].imshow(error_map_axial, alpha=0.5) # Overlay error map
    axes[8].set_title('Errors: FP (Green), FN (Red)')
    axes[8].axis('off')

    
    # 3D View (Optional, can be computationally heavy)
    # from mpl_toolkits.mplot3d import Axes3D
    # ... code to plot 3D surfaces ...
    
    plt.tight_layout()
    #plt.show()





if __name__ == '__main__':
    test_dataset = [
        (load_case('%sa%2d-histeq.nii.gz' % (images_folder, i), '%sa%2d-seg.nii.gz' % (images_folder, i)), '%2d'%(i)) for i in cersegsys_test_prt
    ]

    pred_masks = {
        '%2d'%(i): {} for i in cersegsys_test_prt
    }

    for model_name in model_names:
        for ((original_image, gt_mask), i) in test_dataset:
            print(f'processing {i}')
            pred_mask = nib.load(f'{segmentations_folder}\\{model_name}\\{i}.nii.gz').get_fdata().astype(np.int8)
            
            pred_masks[i][model_name] = pred_mask

            continue
            visualize_failure_case(original_image=original_image, gt_mask=gt_mask, pred_mask=pred_mask)

            

            if not os.path.exists(f'{output_folder}\\{model_name}\\'):
                os.makedirs(f'{output_folder}\\{model_name}\\')
            plt.savefig(f'{output_folder}\\{model_name}\\analysis_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()

    for ((original_image, gt_mask), i) in test_dataset:
        visualize_failure_all(original_image=original_image, gt_mask=gt_mask, pred_masks=pred_masks[i])
        if not os.path.exists(f'{output_folder}\\all\\'):
            os.makedirs(f'{output_folder}\\all\\')

        plt.savefig(f'{output_folder}\\all\\analysis_all_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()


# Example usage:
# for bad_case_id in list_of_worst_case_ids:
#   img, gt, pred = load_case(bad_case_id)
#   visualize_failure_case(img, gt, pred)
#   plt.savefig(f'failure_analysis_{bad_case_id}.png', dpi=300, bbox_inches='tight')
#   plt.close()


#  Analyze and Categorize: For each visualized failure, ask:
#     Is it a registration error? Is the brainstem misaligned in the original pre-processed image?
#     Is it extreme atrophy? Is the structure so small that even a small absolute error leads to a large relative Dice drop?
#     Is it an image artifact? Look for motion blur, noise, or bias field artifacts in the original image.
#     Is it a specific anatomical challenge? e.g., unusually close proximity to another structure.
#  Report Findings: In your paper, dedicate a subsection to "Failure Case Analysis". Include one or two representative figures and discuss the common themes you found. This demonstrates critical thinking and a thorough understanding of your model's limitations.
