import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from skimage.measure import label
from keras_contrib.layers import InstanceNormalization
from keras.models import load_model
from utils.losses import soft_dice_loss, soft_dice_score
from utils.utils import cersegsys_test_prt, cersegsys_train_prt
from visualizations_for_papers import BrainstemVisualizer

class BrainstemSegmentationEvaluator:
    def __init__(self):
        self.LABEL_NAMES = {
            1: 'Medulla_Oblongata',
            2: 'Pons', 
            3: 'Mesencephalon'
        }

    def generate_visualizations(self, image, y_true, y_pred, case_id, output_dir):
        """Generate all visualizations for a case"""
        visualizer = BrainstemVisualizer()
        visualizer.visualize_case(image, y_true, y_pred, case_id, output_dir)
        
    def evaluate_case(self, y_true, y_pred):
        """Calculate all metrics for a single case"""
        metrics = {}
        
        # Per-label metrics
        for label_val, label_name in self.LABEL_NAMES.items():
            mask_true = (y_true == label_val)
            mask_pred = (y_pred == label_val)
            metrics.update(self._calculate_all_metrics(mask_true, mask_pred, label_name))
        
        # Combined brainstem metrics
        bs_true = (y_true > 0)
        bs_pred = (y_pred > 0)
        metrics.update(self._calculate_all_metrics(bs_true, bs_pred, 'Brainstem'))
        
        return metrics
    
    def _calculate_all_metrics(self, y_true, y_pred, prefix):
        """Calculate comprehensive metrics for binary segmentation"""
        metrics = {}
        
        # Basic voxel-based metrics
        metrics.update(self._voxel_metrics(y_true, y_pred, prefix))
        
        # Surface distance metrics
        metrics.update(self._surface_metrics(y_true, y_pred, prefix))
        
        # Volume metrics
        metrics.update(self._volume_metrics(y_true, y_pred, prefix))
        
        # Detection metrics
        metrics.update(self._detection_metrics(y_true, y_pred, prefix))
        
        return metrics
    
    def _voxel_metrics(self, y_true, y_pred, prefix):
        """Traditional segmentation metrics"""
        tp = np.sum(y_true & y_pred)
        fp = np.sum(y_pred & ~y_true)
        fn = np.sum(y_true & ~y_pred)
        tn = np.sum(~y_true & ~y_pred)
        
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        sensitivity = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        
        return {
            f'{prefix}_Dice': dice,
            f'{prefix}_IoU': iou,
            f'{prefix}_Sensitivity': sensitivity,
            f'{prefix}_Specificity': specificity,
            f'{prefix}_Precision': precision
        }
    
    def _surface_metrics(self, y_true, y_pred, prefix):
        """Surface distance metrics"""
        try:
            surface_true = np.argwhere(y_true)
            surface_pred = np.argwhere(y_pred)
            
            if len(surface_true) == 0 or len(surface_pred) == 0:
                return {
                    f'{prefix}_HD': np.nan,
                    f'{prefix}_ASD': np.nan,
                    f'{prefix}_NSD': np.nan
                }
            
            # Hausdorff Distance
            hd = max(directed_hausdorff(surface_true, surface_pred)[0],
                      directed_hausdorff(surface_pred, surface_true)[0])
            
            # Average Surface Distance
            tree_pred = cKDTree(surface_pred)
            dist_true_to_pred = tree_pred.query(surface_true)[0].mean()
            tree_true = cKDTree(surface_true)
            dist_pred_to_true = tree_true.query(surface_pred)[0].mean()
            asd = (dist_true_to_pred + dist_pred_to_true) / 2
            
            # Normalized Surface Dice (1mm tolerance)
            nsd = ((tree_pred.query(surface_true)[0] <= 1.0).mean() + 
                   (tree_true.query(surface_pred)[0] <= 1.0).mean()) / 2
            
            return {
                f'{prefix}_HD': hd,
                f'{prefix}_ASD': asd,
                f'{prefix}_NSD': nsd
            }
        except:
            return {
                f'{prefix}_HD': np.nan,
                f'{prefix}_ASD': np.nan,
                f'{prefix}_NSD': np.nan
            }
    
    def _volume_metrics(self, y_true, y_pred, prefix):
        """Volume-based metrics"""
        vol_true = np.sum(y_true)
        vol_pred = np.sum(y_pred)
        rvd = (vol_pred - vol_true) / (vol_true + 1e-7)
        vs = 1 - abs(vol_true - vol_pred) / (vol_true + vol_pred + 1e-7)
        
        return {
            f'{prefix}_RVD': rvd,
            f'{prefix}_VS': vs,
            f'{prefix}_Vol_True': vol_true,
            f'{prefix}_Vol_Pred': vol_pred
        }
    
    def _detection_metrics(self, y_true, y_pred, prefix):
        """Detection metrics"""
        # Number of connected components
        n_comp_true = label(y_true).max()
        n_comp_pred = label(y_pred).max()
        
        # Detection success (for small structures)
        detected = 1 if (n_comp_pred > 0 and n_comp_true > 0) else 0
        
        return {
            f'{prefix}_Components_True': n_comp_true,
            f'{prefix}_Components_Pred': n_comp_pred,
            f'{prefix}_Detected': detected
        }
    
    def evaluate_dataset(self, model, dataset):
        """Evaluate model on entire dataset"""
        all_metrics = []
        
        for (x, y_true) in tqdm(dataset, desc='Evaluating'):
            y_pred = model.predict(x[None, None, ...])
            
            y_pred = y_pred.squeeze()

            #output = np.zeros((y_pred.shape[1:]))
            output = np.argmax(y_pred > 0.8, axis=0).astype(np.uint8)
            #for indx, mask in enumerate(y_pred):
            #    output += (mask > .5).astype(np.uint8) * indx

            #print(output.shape)
            #y_pred = np.argmax(y_pred, axis=-1)[0]
            case_metrics = self.evaluate_case(y_true, output)
            all_metrics.append(case_metrics)
        
        return pd.DataFrame(all_metrics)
    
    def generate_report(self, df_metrics, output_dir):
        """Generate comprehensive report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate statistics
        stats = {}
        for col in df_metrics.columns:
            if df_metrics[col].dtype in [np.float64, np.int64]:
                stats[f'{col}_mean'] = df_metrics[col].mean()
                stats[f'{col}_std'] = df_metrics[col].std()
                stats[f'{col}_median'] = df_metrics[col].median()
                stats[f'{col}_q25'] = df_metrics[col].quantile(0.25)
                stats[f'{col}_q75'] = df_metrics[col].quantile(0.75)
        
        # Save reports
        df_metrics.to_csv(os.path.join(output_dir, 'casewise_metrics.csv'), index=False)
        pd.DataFrame([stats]).to_csv(os.path.join(output_dir, 'aggregated_metrics.csv'), index=False)
        
        # Generate LaTeX table
        self._generate_latex_table(stats, output_dir)
        
        return stats
    
    def _generate_latex_table(self, stats, output_dir):
        """Generate publication-ready LaTeX table"""
        latex_lines = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{Segmentation Performance Metrics}",
            "\\label{tab:metrics}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Metric & Mean & Std & Median & Q25 & Q75 \\\\",
            "\\midrule"
        ]
        
        # Add rows for each metric group
        for structure in list(self.LABEL_NAMES.values()) + ['Brainstem']:
            latex_lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{structure}}}}} \\\\")
            
            for metric in ['Dice', 'IoU', 'HD', 'ASD', 'NSD', 'RVD']:
                key = f'{structure}_{metric}_mean'
                if key in stats:
                    row = [
                        f"  {metric}",
                        f"{stats[key]:.3f}",
                        f"{stats[f'{structure}_{metric}_std']:.3f}",
                        f"{stats[f'{structure}_{metric}_median']:.3f}",
                        f"{stats[f'{structure}_{metric}_q25']:.3f}",
                        f"{stats[f'{structure}_{metric}_q75']:.3f}"
                    ]
                    latex_lines.append(" & ".join(row) + " \\\\")
        
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        with open(os.path.join(output_dir, 'metrics_table.tex'), 'w') as f:
            f.write("\n".join(latex_lines))


def load_case(image_path, label_path):
    """Load a single case from NIfTI files"""
    img = nib.load(image_path).get_fdata().astype(np.float32)
    lbl = nib.load(label_path).get_fdata().astype(np.int8)
    
    # Normalize image (example using 0-1 normalization)
    img = (img - img.min()) / (img.max() - img.min())
    
    return img, lbl

models = {
    'unet': 'D:\\university\\phd\\phd\\weights\\mipaim_unet\\202504_only_unet_v2_br\\seg\\model.epoch=165.val_dice_score=0.99581.h5',
    'acapulco': 'D:\\university\\phd\\phd\\weights\\mipaim_unet\\202504_acapulco_br\\seg\\model.epoch=165.val_dice_score=0.99495.h5',
    'mipaim': 'D:\\university\\phd\\phd\\weights\\mipaim_unet\\202504_br\\seg\\model.epoch=165.val_dice_score=0.99154.h5'
}
images_folder = 'D:\\university\\phd\\phd\\datasets\\cersegsys_7\\data\\'


def perform_test():
    test_dataset = [
        load_case('%sa%2d-histeq.nii.gz' % (images_folder, i), '%sa%2d-seg.nii.gz' % (images_folder, i)) for i in cersegsys_train_prt
    ]

    

    for nodel_name in models:
        print(nodel_name)
        model_ = load_model(models[nodel_name], custom_objects={'soft_dice_score': soft_dice_score, 'soft_dice_loss': soft_dice_loss, 'InstanceNormalization': InstanceNormalization})
        # Initialize evaluator
        evaluator = BrainstemSegmentationEvaluator()

        # Evaluate model on test set
        test_metrics = evaluator.evaluate_dataset(model_, test_dataset)

        # Generate comprehensive report
        stats = evaluator.generate_report(test_metrics, f'evals/paper_metrics/train_set/{nodel_name}')

        # Print key results
        print(f"Mean Brainstem Dice: {stats['Brainstem_Dice_mean']:.3f} ± {stats['Brainstem_Dice_std']:.3f}")
        print(f"Mean Medula Dice: {stats['Medulla_Oblongata_Dice_mean']:.3f} ± {stats['Medulla_Oblongata_Dice_std']:.3f}")
        print(f"Mean Pons Dice: {stats['Pons_Dice_mean']:.3f} ± {stats['Pons_Dice_std']:.3f}")
        print(f"Mean Mesencephalon Dice: {stats['Mesencephalon_Dice_mean']:.3f} ± {stats['Mesencephalon_Dice_std']:.3f}")
        print(f"Median HD: {stats['Mesencephalon_HD_median']:.2f} mm")



def make_visualizations():
    # Initialize evaluator and visualizer
    #evaluator = BrainstemSegmentationEvaluator()
    visualizer = BrainstemVisualizer()

    test_dataset = [
        load_case('%sa%2d-histeq.nii.gz' % (images_folder, i), '%sa%2d-seg.nii.gz' % (images_folder, i)) for i in cersegsys_test_prt
    ]
    
    # For a test case

    for nodel_name in models:
        print(nodel_name)
        model_ = load_model(models[nodel_name], custom_objects={'soft_dice_score': soft_dice_score, 'soft_dice_loss': soft_dice_loss, 'InstanceNormalization': InstanceNormalization})

        image, y_true = test_dataset[5]  # Load your test case
        y_pred = model_.predict(image[None, None, ...])
        y_pred = y_pred.squeeze()

        #output = np.zeros((y_pred.shape[1:]))
        output = np.argmax(y_pred > 0.8, axis=0).astype(np.uint8)

        # Generate metrics and visualizations
        #metrics = evaluator.evaluate_case(y_true, y_pred)
        visualizer.visualize_case(image, y_true, output, 
                                "case_001", f'evals/paper_metrics/test_set/{nodel_name}/visualizations/')

        # For paper figures - highlight a representative case
        visualizer.visualize_case(image, y_true, output,
                                "representative_case", f'evals/paper_metrics/test_set/{nodel_name}/figures')
if __name__ == '__main__':
    #perform_test()
    make_visualizations()
    
