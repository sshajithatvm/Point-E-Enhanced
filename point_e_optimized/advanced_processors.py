import open3d as o3d
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from scipy.spatial import KDTree, Delaunay
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import logging
from .validator import PointCloudValidator

class AdvancedPointCloudProcessor:
    """Advanced point cloud processor with density-preserving enhancement."""
    
    def __init__(self, num_workers: int = mp.cpu_count()):
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self.validator = PointCloudValidator()
    
    def enhance_batch(self, point_clouds: List[object]) -> List[object]:
        """Enhance multiple point clouds with density preservation."""
        # Validate input point clouds first
        for i, pc in enumerate(point_clouds):
            validation_result = self.validator.validate_point_cloud(pc, min_points=50)
            if not validation_result.is_valid:
                self.logger.error(f"Input point cloud {i} failed validation: {validation_result.errors}")
                continue
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._enhance_single_advanced, pc) for pc in point_clouds]
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
    
    def _enhance_single_advanced(self, pc) -> object:
        """Enhance single point cloud with density-preserving techniques."""
        points = np.array(pc.coords)
        original_count = len(points)
        
        # Pre-validation
        if len(points) < 50:
            self.logger.warning(f"Point cloud too small for enhancement: {len(points)} points")
            return pc
        
        # Step 1: Advanced outlier removal (preserves density)
        cleaned_points = self._adaptive_outlier_removal(points)
        
        # Step 2: Surface reconstruction with Poisson
        mesh_points = self._poisson_surface_reconstruction(cleaned_points)
        
        # Step 3: Adaptive upsampling for density preservation
        upsampled_points = self._adaptive_upsampling(mesh_points, target_count=original_count)
        
        # Step 4: Advanced smoothing with edge preservation
        smoothed_points = self._edge_preserving_smoothing(upsampled_points)
        
        # Step 5: Density optimization (preserves or increases count)
        final_points = self._density_preserving_optimization(smoothed_points, original_count)
        
        # Final validation
        if len(final_points) < 50:
            self.logger.warning(f"Final point cloud too small: {len(final_points)} points")
            return self._create_point_cloud(points, pc.channels)
        
        # Create enhanced point cloud
        enhanced_pc = self._create_point_cloud(final_points, pc.channels)
        
        return enhanced_pc
    
    def _adaptive_outlier_removal(self, points: np.ndarray) -> np.ndarray:
        """Adaptive outlier removal that preserves density."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate local density for adaptive parameters
        tree = KDTree(points)
        distances, _ = tree.query(points, k=6)
        local_density = 1.0 / (np.mean(distances, axis=1) + 1e-8)
        
        # Adaptive statistical outlier removal
        std_ratio = max(1.0, min(3.0, np.percentile(local_density, 75) / np.percentile(local_density, 25)))
        nb_neighbors = min(30, max(10, int(len(points) * 0.02)))
        
        pcd_cleaned, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        # Adaptive radius outlier removal
        radius = np.percentile(distances[:, 1], 50) * 2  # Median distance * 2
        pcd_cleaned, _ = pcd_cleaned.remove_radius_outlier(nb_points=5, radius=radius)
        
        return np.asarray(pcd_cleaned.points)
    
    def _poisson_surface_reconstruction(self, points: np.ndarray) -> np.ndarray:
        """Poisson surface reconstruction for smooth geometry."""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Estimate normals for reconstruction
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            # Orient normals consistently
            pcd.orient_normals_consistent_tangent_plane(100)
            
            # Poisson surface reconstruction
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8, width=0, scale=1.1, linear_fit=False
                )
            
            # Sample points from mesh for smooth surface
            mesh_points = mesh.sample_points_uniformly(number_of_points=len(points) * 2)
            return np.asarray(mesh_points.points)
            
        except Exception as e:
            self.logger.warning(f"Poisson reconstruction failed: {e}, using original points")
            return points
    
    def _adaptive_upsampling(self, points: np.ndarray, target_count: int) -> np.ndarray:
        """Adaptive upsampling to reach or exceed target density."""
        current_count = len(points)
        
        if current_count >= target_count:
            return points
        
        # Calculate upsampling factor
        upsampling_factor = target_count / current_count
        
        # Use Delaunay triangulation for intelligent upsampling
        try:
            if len(points) >= 4:  # Minimum for 3D Delaunay
                # Project to 2D for triangulation (use XY plane)
                points_2d = points[:, :2]
                tri = Delaunay(points_2d)
                
                # Generate new points on triangle centers
                new_points = []
                for simplex in tri.simplices:
                    if np.random.random() < min(0.3, upsampling_factor - 1.0):
                        # Interpolate point on triangle
                        triangle_points = points[simplex]
                        center = np.mean(triangle_points, axis=0)
                        
                        # Add slight random offset for natural variation
                        offset = np.random.randn(3) * 0.001
                        new_points.append(center + offset)
                
                if new_points:
                    upsampled = np.vstack([points, np.array(new_points)])
                    return upsampled[:target_count]  # Limit to target count
            
        except Exception as e:
            self.logger.warning(f"Delaunay upsampling failed: {e}")
        
        # Fallback: simple interpolation
        return self._interpolation_upsampling(points, target_count)
    
    def _interpolation_upsampling(self, points: np.ndarray, target_count: int) -> np.ndarray:
        """Simple interpolation-based upsampling."""
        current_count = len(points)
        
        if current_count >= target_count:
            return points
        
        # Create interpolation grid
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        # Generate grid points
        grid_factor = int(np.ceil(target_count ** (1/3) * 1.2))
        xi = np.linspace(x_min, x_max, grid_factor)
        yi = np.linspace(y_min, y_max, grid_factor)
        zi = np.linspace(z_min, z_max, grid_factor)
        
        # Interpolate
        try:
            grid_points = np.array(np.meshgrid(xi, yi, zi, indexing='ij')).T.reshape(-1, 3)
            
            # Use nearest neighbor interpolation for Z
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=3).fit(points)
            distances, indices = nbrs.kneighbors(grid_points)
            
            # Weighted average based on distance
            weights = 1.0 / (distances + 1e-8)
            weights /= weights.sum(axis=1, keepdims=True)
            
            interpolated = np.sum(points[indices] * weights[:, :, np.newaxis], axis=1)
            
            # Combine with original points
            combined = np.vstack([points, interpolated])
            return combined[:target_count]
            
        except Exception as e:
            self.logger.warning(f"Interpolation upsampling failed: {e}")
            return points
    
    def _edge_preserving_smoothing(self, points: np.ndarray) -> np.ndarray:
        """Edge-preserving bilateral smoothing."""
        try:
            tree = KDTree(points)
            smoothed = np.zeros_like(points)
            
            for i in range(len(points)):
                # Find neighbors
                distances, indices = tree.query(points[i], k=min(20, len(points)))
                neighbors = points[indices]
                
                # Bilateral weights
                spatial_weights = np.exp(-distances**2 / (2 * 0.01**2))
                
                # Intensity weights (based on Z coordinate)
                intensity_diff = np.abs(neighbors[:, 2] - points[i, 2])
                intensity_weights = np.exp(-intensity_diff**2 / (2 * 0.05**2))
                
                # Combined weights
                weights = spatial_weights * intensity_weights
                weights /= weights.sum()
                
                # Smooth point
                smoothed[i] = np.sum(neighbors * weights[:, np.newaxis], axis=0)
            
            return smoothed
            
        except Exception as e:
            self.logger.warning(f"Edge-preserving smoothing failed: {e}")
            return gaussian_filter(points, sigma=0.5, mode='reflect')
    
    def _density_preserving_optimization(self, points: np.ndarray, original_count: int) -> np.ndarray:
        """Optimize density while preserving or increasing point count."""
        current_count = len(points)
        
        # Ensure we don't reduce density
        if current_count < original_count:
            # Upsample to original count
            return self._adaptive_upsampling(points, original_count)
        elif current_count > original_count * 1.5:
            # Only downsample if significantly over target
            voxel_size = (current_count / original_count) ** 0.33 * 0.01
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
            result = np.asarray(downsampled.points)
            
            # Ensure minimum count
            if len(result) < original_count:
                return self._adaptive_upsampling(result, original_count)
            return result
        else:
            # Keep current density (within 50% of original)
            return points
    
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
