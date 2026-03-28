import open3d as o3d
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter, median_filter
import logging
from .validator import PointCloudValidator

class AggressiveDetailProcessor:
    """Aggressive detail processor for clearly visible enhancement with sharp edges."""
    
    def __init__(self, num_workers: int = mp.cpu_count()):
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self.validator = PointCloudValidator()
    
    def enhance_batch(self, point_clouds: List[object]) -> List[object]:
        """Enhance multiple point clouds with aggressive detail improvement."""
        # Validate input point clouds first
        for i, pc in enumerate(point_clouds):
            validation_result = self.validator.validate_point_cloud(pc, min_points=50)
            if not validation_result.is_valid:
                self.logger.error(f"Input point cloud {i} failed validation: {validation_result.errors}")
                continue
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._enhance_aggressive_detail, pc) for pc in point_clouds]
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
    
    def _enhance_aggressive_detail(self, pc) -> object:
        """Aggressive enhancement with clearly visible improvements."""
        try:
            points = np.array(pc.coords)
        except Exception as e:
            self.logger.error(f"Failed to extract points from point cloud: {e}")
            return pc
            
        original_count = len(points)
        
        # Pre-validation
        if original_count == 0:
            self.logger.warning("Empty point cloud - skipping enhancement")
            return pc
        elif original_count < 50:
            self.logger.warning(f"Point cloud too small for enhancement: {original_count} points")
            return pc
        
        # Step 1: Preserve all original points
        preserved_points = points.copy()
        
        # Step 2: Advanced edge and curvature detection
        edge_map, curvature_map, normal_map = self._advanced_edge_detection(preserved_points)
        
        # Step 3: Aggressive edge sharpening
        sharpened_points = self._aggressive_edge_sharpening(preserved_points, edge_map, normal_map)
        
        # Step 4: Surface continuity improvement
        continuity_points = self._surface_continuity_improvement(sharpened_points, edge_map, curvature_map)
        
        # Step 5: High-density edge-focused upsampling
        upsampled_points = self._edge_focused_upsampling(continuity_points, edge_map, curvature_map, original_count)
        
        # Step 6: Final detail enhancement
        final_points = self._final_detail_enhancement(upsampled_points, edge_map, curvature_map)
        
        # Create enhanced point cloud
        enhanced_pc = self._create_point_cloud(final_points, pc.channels)
        
        return enhanced_pc
    
    def _advanced_edge_detection(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advanced edge, curvature, and normal detection for aggressive enhancement."""
        try:
            tree = KDTree(points)
            edge_map = np.zeros(len(points))
            curvature_map = np.zeros(len(points))
            normal_map = np.zeros((len(points), 3))
            
            for i in range(len(points)):
                # Find larger neighborhood for better analysis
                distances, indices = tree.query(points[i], k=min(25, len(points)))
                neighbors = points[indices]
                
                if len(neighbors) >= 10:
                    # Advanced PCA for surface analysis
                    centered = neighbors - np.mean(neighbors, axis=0)
                    cov_matrix = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                    
                    # Sort eigenvalues and eigenvectors
                    idx = np.argsort(eigenvalues)
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    
                    # Enhanced edge detection using multiple criteria
                    if eigenvalues[2] > 1e-8 and len(eigenvectors) >= 3:
                        # Planarity (surface vs. line/point)
                        planarity = eigenvalues[0] / eigenvalues[2]
                        linearity = eigenvalues[1] / eigenvalues[2]
                        sphericity = eigenvalues[0] / eigenvalues[1]
                        
                        # Combined edge strength
                        edge_strength = (1.0 - planarity) * 0.6 + (1.0 - linearity) * 0.3 + (1.0 - sphericity) * 0.1
                        edge_map[i] = np.clip(edge_strength, 0, 1)
                        
                        # Enhanced curvature estimation
                        curvature_map[i] = (eigenvalues[0] + eigenvalues[1]) / eigenvalues[2]
                        
                        # Normal vector (smallest eigenvalue direction)
                        normal_map[i] = eigenvectors[:, 0]
                    else:
                        edge_map[i] = 0.0
                        curvature_map[i] = 0.0
                        normal_map[i] = np.array([0, 0, 1])
                else:
                    edge_map[i] = 0.0
                    curvature_map[i] = 0.0
                    normal_map[i] = np.array([0, 0, 1])
            
            # Normalize maps
            edge_map = np.clip(edge_map, 0, 1)
            curvature_map = np.clip(curvature_map, 0, 1)
            
            return edge_map, curvature_map, normal_map
            
        except Exception as e:
            self.logger.warning(f"Advanced edge detection failed: {e}")
            return np.zeros(len(points)), np.zeros(len(points)), np.zeros((len(points), 3))
    
    def _aggressive_edge_sharpening(self, points: np.ndarray, edge_map: np.ndarray, normal_map: np.ndarray) -> np.ndarray:
        """Aggressive edge sharpening for clearly visible improvement."""
        try:
            tree = KDTree(points)
            sharpened = points.copy()
            
            for i in range(len(points)):
                edge_strength = edge_map[i]
                normal = normal_map[i]
                
                if edge_strength > 0.2:  # Only process edge regions
                    # Find neighbors for local analysis
                    distances, indices = tree.query(points[i], k=min(20, len(points)))
                    neighbors = points[indices]
                    
                    # Aggressive sharpening based on edge strength
                    if edge_strength > 0.7:  # Strong edge
                        sharpen_factor = 1.4
                        neighborhood_size = 8
                    elif edge_strength > 0.5:  # Medium edge
                        sharpen_factor = 1.25
                        neighborhood_size = 12
                    else:  # Weak edge
                        sharpen_factor = 1.15
                        neighborhood_size = 16
                    
                    # Local surface fitting
                    local_neighbors = neighbors[:neighborhood_size]
                    if len(local_neighbors) >= 4:
                        # Fit local plane
                        local_center = np.mean(local_neighbors, axis=0)
                        local_centered = local_neighbors - local_center
                        local_cov = np.cov(local_centered.T)
                        local_eigenvalues, local_eigenvectors = np.linalg.eigh(local_cov)
                        
                        # Sharpen along normal direction
                        normal_contribution = local_eigenvectors[:, 0]
                        sharpening_offset = normal_contribution * (sharpen_factor - 1.0) * 0.01
                        
                        # Apply aggressive sharpening
                        sharpened[i] = points[i] + sharpening_offset
            
            return sharpened
            
        except Exception as e:
            self.logger.warning(f"Aggressive edge sharpening failed: {e}")
            return points
    
    def _surface_continuity_improvement(self, points: np.ndarray, edge_map: np.ndarray, curvature_map: np.ndarray) -> np.ndarray:
        """Improve surface continuity with adaptive processing."""
        try:
            tree = KDTree(points)
            improved = points.copy()
            
            for i in range(len(points)):
                edge_strength = edge_map[i]
                curvature = curvature_map[i]
                
                # Different processing based on local geometry
                if edge_strength > 0.6:  # Edge region - preserve sharpness
                    # Minimal processing for edge preservation
                    distances, indices = tree.query(points[i], k=min(10, len(points)))
                    neighbors = points[indices]
                    
                    # Very light smoothing to maintain sharpness
                    weights = np.exp(-distances**2 / (2 * 0.005**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    improved[i] = points[i] * 0.9 + local_mean * 0.1
                    
                elif curvature > 0.4:  # High curvature - enhance continuity
                    # Moderate processing for curved surfaces
                    distances, indices = tree.query(points[i], k=min(15, len(points)))
                    neighbors = points[indices]
                    
                    # Curvature-aware smoothing
                    weights = np.exp(-distances**2 / (2 * 0.008**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    local_var = np.var(neighbors, axis=0)
                    
                    # Enhance based on curvature
                    enhancement_factor = 1.0 + curvature * 0.3
                    improved[i] = local_mean + (points[i] - local_mean) * enhancement_factor
                    
                else:  # Flat area - more aggressive smoothing
                    # Stronger processing for flat areas
                    distances, indices = tree.query(points[i], k=min(25, len(points)))
                    neighbors = points[indices]
                    
                    # Strong smoothing for flat areas
                    weights = np.exp(-distances**2 / (2 * 0.015**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    improved[i] = points[i] * 0.7 + local_mean * 0.3
            
            return improved
            
        except Exception as e:
            self.logger.warning(f"Surface continuity improvement failed: {e}")
            return points
    
    def _edge_focused_upsampling(self, points: np.ndarray, edge_map: np.ndarray, curvature_map: np.ndarray, original_count: int) -> np.ndarray:
        """Aggressive upsampling focused on edges and high-curvature regions."""
        current_count = len(points)
        
        # Target significant increase for visible improvement
        target_increase = int(original_count * 0.35)  # 35% increase
        target_count = original_count + target_increase
        
        if current_count >= target_count:
            return points
        
        # Create enhanced priority map
        priority_map = edge_map * 0.7 + curvature_map * 0.3  # Strong focus on edges
        
        # Find high-priority regions for aggressive point insertion
        tree = KDTree(points)
        insertion_candidates = []
        
        # Sort points by priority for processing
        priority_indices = np.argsort(priority_map)[::-1]  # Highest priority first
        
        for idx in priority_indices:
            if len(insertion_candidates) >= target_increase:
                break
                
            if priority_map[idx] > 0.25:  # Only in interesting regions
                base_point = points[idx]
                
                # Check local density
                distances, _ = tree.query(base_point, k=min(12, len(points)))
                local_density = 1.0 / (np.mean(distances) + 1e-8)
                
                # Aggressive point addition in high-priority, low-density areas
                if local_density < 150 and priority_map[idx] > 0.4:
                    # Number of points to add based on priority
                    points_to_add = min(6, int(priority_map[idx] * 8))
                    
                    for _ in range(points_to_add):
                        if len(insertion_candidates) >= target_increase:
                            break
                        
                        # Create new point with geometric awareness
                        # Larger offset range for more visible improvement
                        offset_range = 0.015 * (2.0 - priority_map[idx])
                        offset = np.random.randn(3) * offset_range
                        
                        # Strong bias towards edge direction
                        if edge_map[idx] > 0.5:
                            # Find edge direction
                            distances, indices = tree.query(base_point, k=min(10, len(points)))
                            neighbors = points[indices]
                            
                            if len(neighbors) >= 4:
                                # Enhanced edge direction estimation
                                centered = neighbors - np.mean(neighbors, axis=0)
                                cov_matrix = np.cov(centered.T)
                                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                                
                                # Use multiple eigenvectors for better edge following
                                edge_direction1 = eigenvectors[:, 1]
                                edge_direction2 = eigenvectors[:, 2]
                                
                                # Combine edge directions
                                combined_direction = edge_direction1 * 0.6 + edge_direction2 * 0.4
                                combined_direction /= np.linalg.norm(combined_direction)
                                
                                # Strong edge bias
                                offset = offset * 0.2 + np.dot(offset, combined_direction) * combined_direction * 0.8
                        
                        new_point = base_point + offset
                        
                        # Ensure minimum distance from existing points
                        distances_check, _ = tree.query(new_point, k=1)
                        if distances_check > 0.002:  # Smaller minimum distance for denser packing
                            insertion_candidates.append(new_point)
        
        if insertion_candidates:
            # Combine original and new points
            combined = np.vstack([points, np.array(insertion_candidates)])
            return combined
        
        return points
    
    def _final_detail_enhancement(self, points: np.ndarray, edge_map: np.ndarray, curvature_map: np.ndarray) -> np.ndarray:
        """Final aggressive detail enhancement."""
        try:
            tree = KDTree(points)
            final = points.copy()
            
            for i in range(len(points)):
                edge_strength = edge_map[i]
                curvature = curvature_map[i]
                
                # Aggressive enhancement based on local geometry
                if edge_strength > 0.5:  # Edge region
                    # Strong edge enhancement
                    distances, indices = tree.query(points[i], k=min(12, len(points)))
                    neighbors = points[indices]
                    
                    # Edge-preserving enhancement
                    weights = np.exp(-distances**2 / (2 * 0.004**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    local_var = np.var(neighbors, axis=0)
                    
                    # Aggressive edge sharpening
                    sharpening_strength = 1.0 + edge_strength * 0.5
                    final[i] = local_mean + (points[i] - local_mean) * sharpening_strength
                    
                elif curvature > 0.3:  # High curvature
                    # Detail enhancement in curved regions
                    distances, indices = tree.query(points[i], k=min(18, len(points)))
                    neighbors = points[indices]
                    
                    # Curvature-aware enhancement
                    weights = np.exp(-distances**2 / (2 * 0.006**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    
                    # Aggressive curvature enhancement
                    detail_factor = 1.0 + curvature * 0.6
                    final[i] = local_mean + (points[i] - local_mean) * detail_factor
                    
                else:  # Flat area
                    # Surface quality improvement
                    distances, indices = tree.query(points[i], k=min(20, len(points)))
                    neighbors = points[indices]
                    
                    # Quality enhancement for flat areas
                    weights = np.exp(-distances**2 / (2 * 0.01**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    final[i] = points[i] * 0.8 + local_mean * 0.2
            
            return final
            
        except Exception as e:
            self.logger.warning(f"Final detail enhancement failed: {e}")
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
                        # Enhanced interpolation for new points
                        enhanced_channels[name] = np.interp(
                            np.linspace(0, 1, len(points)),
                            np.linspace(0, 1, len(values)),
                            values
                        )
                    else:
                        enhanced_channels[name] = np.ones(len(points)) * 0.5
        
        return PointCloud(coords=points, channels=enhanced_channels)
