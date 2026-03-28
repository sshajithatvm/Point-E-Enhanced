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

class GeometryAwareProcessor:
    """Geometry-aware point cloud processor that preserves all points while enhancing clarity and density."""
    
    def __init__(self, num_workers: int = mp.cpu_count()):
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self.validator = PointCloudValidator()
    
    def enhance_batch(self, point_clouds: List[object]) -> List[object]:
        """Enhance multiple point clouds with geometry-aware techniques."""
        # Validate input point clouds first
        for i, pc in enumerate(point_clouds):
            validation_result = self.validator.validate_point_cloud(pc, min_points=50)
            if not validation_result.is_valid:
                self.logger.error(f"Input point cloud {i} failed validation: {validation_result.errors}")
                continue
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._enhance_geometry_aware, pc) for pc in point_clouds]
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
    
    def _enhance_geometry_aware(self, pc) -> object:
        """Geometry-aware enhancement that preserves all points while improving clarity."""
        points = np.array(pc.coords)
        original_count = len(points)
        
        # Pre-validation
        if len(points) < 50:
            self.logger.warning(f"Point cloud too small for enhancement: {len(points)} points")
            return pc
        
        # Step 1: Preserve all original points (no removal)
        preserved_points = points.copy()
        
        # Step 2: Geometry-aware noise reduction (preserves all points)
        noise_reduced_points = self._geometry_aware_noise_reduction(preserved_points)
        
        # Step 3: Edge-preserving smoothing (maintains sharp features)
        smoothed_points = self._edge_preserving_smoothing(noise_reduced_points)
        
        # Step 4: Geometry-aware upsampling (adds points intelligently)
        upsampled_points = self._geometry_aware_upsampling(smoothed_points, original_count)
        
        # Step 5: Density optimization (ensures better distribution)
        final_points = self._density_optimization(upsampled_points, original_count)
        
        # Create enhanced point cloud
        enhanced_pc = self._create_point_cloud(final_points, pc.channels)
        
        return enhanced_pc
    
    def _geometry_aware_noise_reduction(self, points: np.ndarray) -> np.ndarray:
        """Geometry-aware noise reduction that preserves all points."""
        try:
            tree = KDTree(points)
            denoised = np.zeros_like(points)
            
            for i in range(len(points)):
                # Find neighbors for local analysis
                distances, indices = tree.query(points[i], k=min(12, len(points)))
                neighbors = points[indices]
                
                # Calculate local geometric properties
                local_center = np.mean(neighbors, axis=0)
                local_variance = np.var(neighbors, axis=0)
                
                # Geometry-aware filtering strength
                # Stronger filtering in flat areas, weaker near edges
                edge_indicator = np.sum(local_variance)
                if edge_indicator < 0.001:  # Flat area
                    filter_strength = 0.3
                elif edge_indicator < 0.01:  # Gentle variation
                    filter_strength = 0.15
                else:  # Edge/corner area
                    filter_strength = 0.05
                
                # Apply gentle filtering
                weights = np.exp(-distances**2 / (2 * 0.02**2))
                weights /= weights.sum()
                
                local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                denoised[i] = points[i] * (1 - filter_strength) + local_mean * filter_strength
            
            return denoised
            
        except Exception as e:
            self.logger.warning(f"Geometry-aware noise reduction failed: {e}")
            return points
    
    def _edge_preserving_smoothing(self, points: np.ndarray) -> np.ndarray:
        """Edge-preserving smoothing that maintains sharp features."""
        try:
            tree = KDTree(points)
            smoothed = np.zeros_like(points)
            
            for i in range(len(points)):
                # Find neighbors
                distances, indices = tree.query(points[i], k=min(20, len(points)))
                neighbors = points[indices]
                
                # Calculate local normal variation (edge detection)
                if len(neighbors) >= 4:
                    # Estimate local normal using PCA
                    centered = neighbors - np.mean(neighbors, axis=0)
                    cov_matrix = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                    normal = eigenvectors[:, 0]  # Smallest eigenvalue
                    
                    # Check normal consistency
                    neighbor_normals = []
                    for j, neighbor_idx in enumerate(indices[1:], 1):  # Skip self
                        neighbor_points = points[indices[max(0, j-3):j+4]]  # Local window
                        if len(neighbor_points) >= 4:
                            centered_n = neighbor_points - np.mean(neighbor_points, axis=0)
                            cov_n = np.cov(centered_n.T)
                            eigen_n, eigvec_n = np.linalg.eigh(cov_n)
                            neighbor_normals.append(eigvec_n[:, 0])
                    
                    if neighbor_normals:
                        normal_variation = np.std([np.dot(normal, n) for n in neighbor_normals])
                        edge_strength = normal_variation
                    else:
                        edge_strength = 0.0
                else:
                    edge_strength = 0.0
                
                # Adaptive smoothing based on edge strength
                if edge_strength > 0.3:  # Strong edge
                    smooth_weight = 0.02
                elif edge_strength > 0.1:  # Medium edge
                    smooth_weight = 0.08
                else:  # Flat area
                    smooth_weight = 0.2
                
                # Apply smoothing
                spatial_weights = np.exp(-distances**2 / (2 * 0.015**2))
                spatial_weights /= spatial_weights.sum()
                
                local_smooth = np.sum(neighbors * spatial_weights[:, np.newaxis], axis=0)
                smoothed[i] = points[i] * (1 - smooth_weight) + local_smooth * smooth_weight
            
            return smoothed
            
        except Exception as e:
            self.logger.warning(f"Edge-preserving smoothing failed: {e}")
            # Fallback to very light Gaussian smoothing
            return gaussian_filter(points, sigma=0.3, mode='reflect')
    
    def _geometry_aware_upsampling(self, points: np.ndarray, original_count: int) -> np.ndarray:
        """Geometry-aware upsampling that adds points intelligently."""
        current_count = len(points)
        
        # Calculate target upsampling (aim for 20-30% increase)
        target_increase = int(original_count * 0.25)  # 25% increase
        target_count = original_count + target_increase
        
        if current_count >= target_count:
            return points
        
        # Analyze local density to find sparse areas
        tree = KDTree(points)
        local_densities = []
        
        for point in points:
            distances, _ = tree.query(point, k=min(8, len(points)))
            local_density = 1.0 / (np.mean(distances) + 1e-8)
            local_densities.append(local_density)
        
        local_densities = np.array(local_densities)
        
        # Identify sparse regions (bottom 30% density)
        density_threshold = np.percentile(local_densities, 30)
        sparse_indices = np.where(local_densities < density_threshold)[0]
        
        # Geometry-aware point insertion
        new_points = []
        points_added = 0
        max_points_to_add = target_increase
        
        for sparse_idx in sparse_indices:
            if points_added >= max_points_to_add:
                break
            
            base_point = points[sparse_idx]
            
            # Find local neighborhood for geometry analysis
            distances, neighbor_indices = tree.query(base_point, k=min(10, len(points)))
            neighbors = points[neighbor_indices]
            
            if len(neighbors) >= 3:
                # Estimate local surface normal
                centered = neighbors - np.mean(neighbors, axis=0)
                cov_matrix = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # Check if it's a surface-like region (not a line or point)
                if eigenvalues[1] > eigenvalues[0] * 3:  # Surface-like
                    normal = eigenvectors[:, 0]
                    
                    # Create tangent plane
                    # Generate points in tangent plane
                    for _ in range(min(3, max_points_to_add - points_added)):
                        # Random position in local area
                        u = np.random.uniform(-0.01, 0.01)
                        v = np.random.uniform(-0.01, 0.01)
                        
                        # Tangent vectors (perpendicular to normal)
                        if abs(normal[0]) < 0.9:
                            tangent1 = np.cross(normal, [1, 0, 0])
                        else:
                            tangent1 = np.cross(normal, [0, 1, 0])
                        tangent1 /= np.linalg.norm(tangent1)
                        
                        tangent2 = np.cross(normal, tangent1)
                        tangent2 /= np.linalg.norm(tangent2)
                        
                        # New point on tangent plane
                        new_point = base_point + u * tangent1 + v * tangent2
                        
                        # Small normal offset for surface thickness
                        new_point += normal * np.random.uniform(-0.002, 0.002)
                        
                        # Check if point is not too close to existing points
                        distances_check, _ = tree.query(new_point, k=1)
                        if distances_check > 0.005:  # Minimum distance
                            new_points.append(new_point)
                            points_added += 1
        
        if new_points:
            # Combine original and new points
            combined = np.vstack([points, np.array(new_points)])
            return combined
        
        return points
    
    def _density_optimization(self, points: np.ndarray, original_count: int) -> np.ndarray:
        """Optimize density distribution while preserving all points."""
        current_count = len(points)
        
        # Ensure we have at least the original count
        if current_count < original_count:
            self.logger.warning(f"Point count reduced to {current_count}, this shouldn't happen")
            return points
        
        # If we have significantly more points, we can optionally reduce slightly
        # but never below original count
        if current_count > original_count * 1.5:
            # Gentle downsampling to reasonable level
            target_count = max(original_count, int(original_count * 1.3))
            
            # Use uniform sampling to preserve distribution
            indices = np.random.choice(current_count, target_count, replace=False)
            return points[indices]
        
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
                    # For new points, interpolate channel values
                    if len(values) > 0 and len(points) > len(values):
                        # Simple interpolation for new points
                        enhanced_channels[name] = np.interp(
                            np.linspace(0, 1, len(points)),
                            np.linspace(0, 1, len(values)),
                            values
                        )
                    else:
                        enhanced_channels[name] = np.ones(len(points)) * 0.5
        
        return PointCloud(coords=points, channels=enhanced_channels)
