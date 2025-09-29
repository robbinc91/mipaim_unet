import os
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure

class BrainstemVisualizer:
    def __init__(self):

        # TODO: make voxel_size auto
        self.voxel_size = (1, 1, 1)  # (x,y,z) dimensions in mm
        self.scale_bar_length = 10  # mm
        self.scale_bar_thickness = 3
        self.scale_bar_color = '#1f1'#'white'
        self.scale_bar_offset = (0.1, 0.1)  # Relative position from bottom-right

        self.COLOR_MAP = {
            1: (1, 0, 0),   # Medulla - Red
            2: (0, 1, 0),   # Pons - Green
            3: (0, 0, 1)    # Mesencephalon - Blue
        }
        self.OPACITY = 0.5
        self.SLICE_CMAP = plt.cm.gray
        self.OVERLAY_CMAP = ListedColormap(['black', 'red', 'green', 'blue'])
        self.DPI = 300
        self.FIG_SIZE = (12, 4)

    def _add_scale_bar(self, ax, plane='axial'):
        """Add anatomical scale bar to axis"""
        if plane == 'axial':
            pixel_length = self.scale_bar_length / self.voxel_size[0]  # x-dimension
            orientation = 'horizontal'
        elif plane == 'sagittal':
            pixel_length = self.scale_bar_length / self.voxel_size[1]  # y-dimension
            orientation = 'horizontal'
        else:  # coronal
            pixel_length = self.scale_bar_length / self.voxel_size[2]  # z-dimension
            orientation = 'vertical'
        
        # Calculate position
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_pos = xlim[1] - (xlim[1]-xlim[0])*self.scale_bar_offset[0]
        y_pos = ylim[0] + (ylim[1]-ylim[0])*self.scale_bar_offset[1]
        
        if orientation == 'horizontal':
            ax.plot([x_pos - pixel_length, x_pos], 
                    [y_pos, y_pos], 
                    color=self.scale_bar_color,
                    linewidth=self.scale_bar_thickness)
            ax.text(x_pos - pixel_length/2, 
                    y_pos + (ylim[1]-ylim[0])*0.05, 
                    f'{self.scale_bar_length} mm',
                    color=self.scale_bar_color,
                    ha='center', va='bottom',
                    fontsize=8)
        else:
            ax.plot([x_pos, x_pos],
                    [y_pos, y_pos + pixel_length],
                    color=self.scale_bar_color,
                    linewidth=self.scale_bar_thickness)
            ax.text(x_pos + (xlim[1]-xlim[0])*0.05,
                    y_pos + pixel_length/2,
                    f'{self.scale_bar_length} mm',
                    color=self.scale_bar_color,
                    ha='left', va='center',
                    rotation=90,
                    fontsize=8)
        
    def visualize_case(self, image, y_true, y_pred, case_id, output_dir):
        """Generate all visualizations for a single case"""
        os.makedirs(output_dir, exist_ok=True)

        # Original scan
        self._plot_raw_slices(image, f'Raw MRI - {case_id}', os.path.join(output_dir, f'{case_id}_raw_mri.png'))
        
        # 2D Orthogonal Slices
        self._plot_orthogonal_slices(image, y_true, 
                                   f"Ground Truth - {case_id}",
                                   os.path.join(output_dir, f"{case_id}_gt_slices.png"))
        
        self._plot_orthogonal_slices(image, y_pred,
                                   f"Prediction - {case_id}",
                                   os.path.join(output_dir, f"{case_id}_pred_slices.png"))
        
        # 3D Renderings
        self._plot_3d_segmentation(y_true,
                                  f"Ground Truth 3D - {case_id}",
                                  os.path.join(output_dir, f"{case_id}_gt_3d.png"))
        
        self._plot_3d_segmentation(y_pred,
                                  f"Prediction 3D - {case_id}",
                                  os.path.join(output_dir, f"{case_id}_pred_3d.png"))
        
        # Error Maps
        self._plot_error_maps(image, y_true, y_pred, case_id, output_dir)
        
        # Multiplanar Error Visualization
        self._plot_multiplanar_errors(y_true, y_pred, case_id, output_dir)

    def _plot_raw_slices(self, image, title, save_path):
        """Plot MRI slices without any segmentation overlay"""
        fig, axes = plt.subplots(1, 3, figsize=self.FIG_SIZE)

        slices = {
            'Axial': (image.shape[2]//2, 2),
            'Sagittal': (image.shape[0]//2, 0),
            'Coronal': (image.shape[1]//2, 1)
        }

        # Find optimal contrast parameters
        vmin = np.percentile(image, 1)
        vmax = np.percentile(image, 99)

        for ax, (slice_name, (slice_idx, axis)) in zip(axes, slices.items()):
            if axis == 0:
                img_slice = np.take(image, slice_idx, axis=axis)
            else:
                img_slice = np.take(image, slice_idx, axis=axis).T

            ax.imshow(img_slice, cmap=self.SLICE_CMAP, vmin=vmin, vmax=vmax, interpolation='none')
            ax.set_title(f'{slice_name} View')
            ax.axis('off')

            # Add scale bar
            self._add_scale_bar(ax, slice_name.lower())

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        
    def _plot_orthogonal_slices(self, image, segmentation, title, save_path):
        """Plot axial, sagittal and coronal slices with overlay"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        slices = {
            'Axial': (image.shape[2]//2, 2),
            'Sagittal': (image.shape[0]//2, 0),
            'Coronal': (image.shape[1]//2, 1)
        }
        
        for ax, (slice_name, (slice_idx, axis)) in zip(axes, slices.items()):
            if axis == 0:
                img_slice = np.take(image, slice_idx, axis=axis)
                seg_slice = np.take(segmentation, slice_idx, axis=axis)
            elif axis == 1:
                img_slice = np.take(image, slice_idx, axis=axis).T
                seg_slice = np.take(segmentation, slice_idx, axis=axis).T
            else:
                img_slice = np.take(image, slice_idx, axis=axis).T
                seg_slice = np.take(segmentation, slice_idx, axis=axis).T
            
            ax.imshow(img_slice, cmap=self.SLICE_CMAP, interpolation='none')
            ax.imshow(seg_slice, cmap=self.OVERLAY_CMAP, alpha=self.OPACITY, interpolation='none')
            ax.set_title(f"{slice_name} View")
            ax.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_3d_segmentation(self, segmentation, title, save_path):
        """Create 3D surface rendering of segmentation"""
        fig = plt.figure(figsize=(10, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')

        ax.set_facecolor('black')
        
        for label_val, color in self.COLOR_MAP.items():
            mask = (segmentation == label_val).astype(np.int8)
            if np.sum(mask) == 0:
                continue
                
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
            mesh = Poly3DCollection(verts[faces], 
                                   alpha=self.OPACITY, 
                                   facecolor=color,
                                   edgecolor=color)
            ax.add_collection3d(mesh)
        
        ax.set_xlim(0, segmentation.shape[0])
        ax.set_ylim(0, segmentation.shape[1])
        ax.set_zlim(0, segmentation.shape[2])
        ax.set_title(title)
        ax.view_init(elev=25, azim=45)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_maps(self, image, y_true, y_pred, case_id, output_dir):
        """Generate error maps (FP/FN) visualization"""
        error_map = np.zeros_like(y_true)
        error_map[(y_true > 0) & (y_pred == 0)] = 1  # False Negative
        error_map[(y_true == 0) & (y_pred > 0)] = 2  # False Positive
        
        if np.sum(error_map) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        axial_slice = image.shape[2] // 2
        img_slice = np.take(image, axial_slice, axis=2).T
        err_slice = np.take(error_map, axial_slice, axis=2).T
        
        ax.imshow(img_slice, cmap=self.SLICE_CMAP)
        error_cmap = ListedColormap(['black', 'blue', 'red'])  # Black: BG, Blue: FN, Red: FP
        ax.imshow(err_slice, cmap=error_cmap, alpha=0.6)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='False Negative'),
            Patch(facecolor='red', label='False Positive')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"Error Map - {case_id}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{case_id}_error_map.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_multiplanar_errors(self, y_true, y_pred, case_id, output_dir):
        """Create 3-plane error visualization"""
        error_map = np.zeros_like(y_true)
        error_map[(y_true > 0) & (y_pred == 0)] = 1  # FN
        error_map[(y_true == 0) & (y_pred > 0)] = 2  # FP
        
        if np.sum(error_map) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        planes = [
            ('Axial', error_map[:, :, error_map.shape[2]//2]),
            ('Sagittal', error_map[error_map.shape[0]//2, :, :]),
            ('Coronal', error_map[:, error_map.shape[1]//2, :])
        ]
        
        error_cmap = ListedColormap(['black', 'blue', 'red'])
        
        for ax, (plane, slice_data) in zip(axes, planes):
            ax.imshow(slice_data.T, cmap=error_cmap)
            ax.set_title(f"{plane} Error View")
            ax.axis('off')
        
        plt.suptitle(f"Multiplanar Errors - {case_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{case_id}_multiplanar_errors.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    