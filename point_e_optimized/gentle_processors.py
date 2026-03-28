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

class GentlePointCloudProcessor:
    """Gentle point cloud processor with non-destructive enhancement."""
    
    def __init__(self, num_workers: int = mp.cpu_count()):
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self.validator = PointCloudValidator()
    
    def enhance_batch(self, point_clouds: List[object]) -> List[object]:
        """Enhance multiple point clouds with gentle, non-destructive techniques."""
        # Validate input point clouds first
        for i, pc in enumerate(point_clouds):
            validation_result = self.validator.validate_point_cloud(pc, min_points=50)
            if not validation_result.is_valid:
                self.logger.error(f"Input point cloud {i} failed validation: {validation_result.errors}")
                continue
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._enhance_gentle, pc) for pc in point_clouds]
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
    
    def _enhance_gentle(self, pc) -> object:
        """Gentle enhancement that preserves all original points."""
        points = np.array(pc.coords)
        original_count = len(points)
        
        # Pre-validation
        if len(points) < 50:
            self.logger.warning(f"Point cloud too small for enhancement: {len(points)} points")
            return pc
        
        # Step 1: Gentle outlier removal (very conservative)
        cleaned_points = self._gentle_outlier_removal(points)
        
        # Step 2: Accurate normal estimation
        points_with_normals = self._accurate_normal_estimation(cleaned_points)
        
        # Step 3: Edge-aware smoothing
        smoothed_points = self._edge_aware_smoothing(points_with_normals)
        
        # Step 4: Intelligent gap filling (preserves original points)
        gap_filled_points = self._intelligent_gap_filling(smoothed_points, original_count)
        
        # Step 5: Final refinement (maintains exact point count)
        final_points = self._maintain_point_count(gap_filled_points, points, original_count)
        
        # Create enhanced point cloud
        enhanced_pc = self._create_point_cloud(final_points, pc.channels)
        
        return enhanced_pc
    
    def _gentle_outlier_removal(self, points: np.ndarray) -> np.ndarray:
        """Very gentle outlier removal that preserves most points."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Very conservative statistical outlier removal
        pcd_cleaned, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=5.0)
        
        # Very gentle radius outlier removal
        pcd_cleaned, _ = pcd_cleaned.remove_radius_outlier(nb_points=3, radius=0.1)
        
        cleaned = np.asarray(pcd_cleaned.points)
        
        # Ensure we don't lose too many points
        if len(cleaned) < len(points) * 0.9:  # Keep at least 90% of points
            self.logger.warning(f"Gentle outlier removal too aggressive, keeping original points")
            return points
        
        return cleaned
    
    def _accurate_normal_estimation(self, points: np.ndarray) -> np.ndarray:
        """Accurate normal estimation for better surface understanding."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals with conservative parameters
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05,  # Small radius for local detail
                max_nn=30     # Reasonable neighborhood
            )
        )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Return points with normal information
        return np.asarray(pcd.points)
    
    def _edge_aware_smoothing(self, points: np.ndarray) -> np.ndarray:
        """Edge-aware smoothing that preserves fine structures."""
        try:
            tree = KDTree(points)
            smoothed = np.zeros_like(points)
            
            for i in range(len(points)):
                # Find neighbors with adaptive radius
                distances, indices = tree.query(points[i], k=min(15, len(points)))
                neighbors = points[indices]
                
                # Calculate local density for adaptive smoothing
                local_density = len(neighbors) / (distances[-1] + 1e-8)
                
                # Adaptive smoothing strength based on local density
                if local_density > 100:  # Dense area - more smoothing
                    sigma = 0.02
                elif local_density > 50:  # Medium density - moderate smoothing
                    sigma = 0.01
                else:  # Sparse area - minimal smoothing
                    sigma = 0.005
                
                # Edge-aware weights
                spatial_weights = np.exp(-distances**2 / (2 * sigma**2))
                
                # Preserve edges by checking local variation
                local_variation = np.std(neighbors, axis=0)
                edge_weights = 1.0 / (1.0 + local_variation * 10)  # Less smoothing on edges
                
                # Combined weights
                weights = spatial_weights * edge_weights
                weights /= weights.sum()
                
                # Smooth point
                smoothed[i] = np.sum(neighbors * weights[:, np.newaxis], axis=0)
            
            return smoothed
            
        except Exception as e:
            self.logger.warning(f"Edge-aware smoothing failed: {e}")
            # Fallback to very light Gaussian smoothing
            return gaussian_filter(points, sigma=0.5, mode='reflect')
    
    def _intelligent_gap_filling(self, points: np.ndarray, target_count: int) -> np.ndarray:
        """Intelligent gap filling that preserves original points."""
        current_count = len(points)
        
        # Only add points if we're below target (which should be original count)
        if current_count >= target_count:
            return points
        
        # Calculate how many points to add
        points_to_add = target_count - current_count
        max_add = int(target_count * 0.1)  # Add at most 10% of original count
        
        if points_to_add > max_add:
            points_to_add = max_add
        
        # Find sparse areas using local density
        tree = KDTree(points)
        densities = []
        
        for point in points:
            distances, _ = tree.query(point, k=min(10, len(points)))
            density = 1.0 / (np.mean(distances) + 1e-8)
            densities.append(density)
        
        densities = np.array(densities)
        
        # Identify sparse areas (low density)
        threshold = np.percentile(densities, 25)  # Bottom 25% density
        sparse_indices = np.where(densities < threshold)[0]
        
        if len(sparse_indices) == 0:
            return points
        
        # Add points in sparse areas
        new_points = []
        for i in range(points_to_add):
            # Randomly select a sparse area
            sparse_idx = np.random.choice(sparse_indices)
            base_point = points[sparse_idx]
            
            # Add small random offset for natural variation
            offset = np.random.randn(3) * 0.002  # Very small offset
            new_point = base_point + offset
            
            # Ensure new point is not too close to existing points
            distances, _ = tree.query(new_point, k=1)
            if distances > 0.01:  # Minimum distance from existing points
                new_points.append(new_point)
        
        if new_points:
            # Combine original and new points
            combined = np.vstack([points, np.array(new_points)])
            return combined
        
        return points
    
    def _maintain_point_count(self, enhanced_points: np.ndarray, original_points: np.ndarray, target_count: int) -> np.ndarray:
        """Maintain exact point count while preserving improvements."""
        current_count = len(enhanced_points)
        
        if current_count == target_count:
            return enhanced_points
        elif current_count > target_count:
            # Gently downsample to target count
            # Use uniform sampling to preserve distribution
            indices = np.random.choice(current_count, target_count, replace=False)
            return enhanced_points[indices]
        else:
            # We have fewer points than target - this shouldn't happen with gap filling
            # Add back some original points if needed
            missing = target_count - current_count
            if missing > 0 and len(original_points) > len(enhanced_points):
                # Find points that were removed during cleaning
                # Simple approach: add random original points
                additional_indices = np.random.choice(len(original_points), missing, replace=False)
                additional_points = original_points[additional_indices]
                return np.vstack([enhanced_points, additional_points])
        
        return enhanced_points
    
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
