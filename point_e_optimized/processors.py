import open3d as o3d
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import logging
from .validator import PointCloudValidator

class PointCloudProcessor:
    """High-performance point cloud processor with Open3D optimizations."""
    
    def __init__(self, num_workers: int = mp.cpu_count()):
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self.validator = PointCloudValidator()
    
    def enhance_batch(self, point_clouds: List[object]) -> List[object]:
        """Enhance multiple point clouds in parallel."""
        # Validate input point clouds first
        for i, pc in enumerate(point_clouds):
            validation_result = self.validator.validate_point_cloud(pc, min_points=50)
            if not validation_result.is_valid:
                self.logger.error(f"Input point cloud {i} failed validation: {validation_result.errors}")
                # Skip invalid point clouds
                continue
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._enhance_single, pc) for pc in point_clouds]
            results = []
            for i, future in enumerate(futures):
                try:
                    enhanced_pc = future.result()
                    # Validate enhanced point cloud
                    validation_result = self.validator.validate_point_cloud(enhanced_pc, min_points=100)
                    if validation_result.is_valid:
                        results.append(enhanced_pc)
                        self.logger.info(f"Enhanced point cloud {i} passed validation")
                    else:
                        self.logger.error(f"Enhanced point cloud {i} failed validation: {validation_result.errors}")
                        # Return original if enhancement failed
                        results.append(point_clouds[i])
                except Exception as e:
                    self.logger.error(f"Failed to enhance point cloud {i}: {e}")
                    results.append(point_clouds[i])  # Return original on failure
        
        return results
    
    def _enhance_single(self, pc) -> object:
        """Enhance single point cloud with optimized processing."""
        points = np.array(pc.coords)
        original_count = len(points)
        
        # Pre-validation
        if len(points) < 50:
            self.logger.warning(f"Point cloud too small for enhancement: {len(points)} points")
            return pc
        
        # Optimized outlier removal
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Radius outlier removal
        pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        
        # Check if we still have enough points
        points_array = np.asarray(pcd.points)
        if len(points_array) < 100:
            self.logger.warning(f"Too few points after outlier removal: {len(points_array)}")
            # Return original with minimal processing
            return self._create_point_cloud(points, pc.channels)
        
        # Advanced denoising
        denoised_points = self._adaptive_denoising(points_array)
        
        # Surface reconstruction
        smoothed_points = self._surface_smoothing(denoised_points)
        
        # Density optimization
        final_points = self._optimize_density(smoothed_points)
        
        # Final validation
        if isinstance(final_points, np.ndarray) and len(final_points) < 50:
            self.logger.warning(f"Final point cloud too small: {len(final_points)} points")
            return self._create_point_cloud(points, pc.channels)
        
        # Create enhanced point cloud
        enhanced_pc = self._create_point_cloud(final_points, pc.channels)
        
        return enhanced_pc
    
    def _adaptive_denoising(self, points: np.ndarray) -> np.ndarray:
        """Optimized adaptive denoising using KDTree."""
        tree = KDTree(points)
        denoised = np.zeros_like(points)
        
        for i in range(len(points)):
            distances, indices = tree.query(points[i], k=10)
            weights = np.exp(-distances**2 / (2 * 0.01**2))
            weights /= weights.sum()
            denoised[i] = np.average(points[indices], weights=weights, axis=0)
        
        return denoised
    
    def _surface_smoothing(self, points: np.ndarray) -> np.ndarray:
        """Optimized Laplacian smoothing."""
        smoothed = gaussian_filter(points, sigma=0.5, mode='reflect')
        return smoothed
    
    def _optimize_density(self, points: np.ndarray) -> np.ndarray:
        """Intelligent density optimization."""
        if len(points) > 4096:
            voxel_size = 0.008
        elif len(points) > 2048:
            voxel_size = 0.01
        else:
            voxel_size = 0.015
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        result = np.asarray(downsampled.points)
        
        # Ensure we always return a valid array
        if len(result) == 0:
            self.logger.warning("Voxel downsampling resulted in empty point cloud, returning original")
            return points
        
        return result
    
    def _create_point_cloud(self, points: np.ndarray, channels: Dict) -> object:
        """Create enhanced point cloud object."""
        from point_e.util.point_cloud import PointCloud
        
        enhanced_channels = {}
        if channels:
            for name, values in channels.items():
                if len(values) == len(points):
                    enhanced_channels[name] = values
                else:
                    enhanced_channels[name] = np.ones(len(points)) * 0.5
        
        return PointCloud(coords=points, channels=enhanced_channels)
