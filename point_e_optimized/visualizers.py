import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from typing import List, Dict, Tuple
import logging

class PointCloudVisualizer:
    """High-performance point cloud visualization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_before_after(self, original, enhanced, title: str) -> plt.Figure:
        """Create before/after comparison."""
        fig = plt.figure(figsize=(16, 8))
        
        # Original
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_point_cloud(ax1, original, f"BEFORE\n{len(original.coords)} points")
        
        # Enhanced
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_point_cloud(ax2, enhanced, f"AFTER\n{len(enhanced.coords)} points")
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_point_cloud(self, ax, pc, title: str):
        """Plot point cloud with optimizations."""
        points = np.array(pc.coords)
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
        ax.set_title(title, fontweight='bold')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
