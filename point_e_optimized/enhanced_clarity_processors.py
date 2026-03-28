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

class EnhancedClarityProcessor:
    """Enhanced clarity processor focusing on edge sharpness and structural detail improvement."""
    
    def __init__(self, num_workers: int = mp.cpu_count()):
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self.validator = PointCloudValidator()
    
    def enhance_batch(self, point_clouds: List[object]) -> List[object]:
        """Enhance multiple point clouds with improved clarity and sharpness."""
        # Validate input point clouds first
        for i, pc in enumerate(point_clouds):
            validation_result = self.validator.validate_point_cloud(pc, min_points=50)
            if not validation_result.is_valid:
                self.logger.error(f"Input point cloud {i} failed validation: {validation_result.errors}")
                continue
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._enhance_clarity, pc) for pc in point_clouds]
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
    
    def _enhance_clarity(self, pc) -> object:
        """Enhance point cloud clarity with improved edge sharpness and structural detail."""
        points = np.array(pc.coords)
        original_count = len(points)
        
        # Pre-validation
        if len(points) < 50:
            self.logger.warning(f"Point cloud too small for enhancement: {len(points)} points")
            return pc
        
        # Step 1: Preserve all original points
        preserved_points = points.copy()
        
        # Step 2: Advanced geometric analysis
        edge_map, curvature_map, normal_map, density_map = self._advanced_geometric_analysis(preserved_points)
        
        # Step 3: Intelligent noise reduction
        noise_reduced_points = self._intelligent_noise_reduction(preserved_points, edge_map, density_map)
        
        # Step 4: Edge sharpening and boundary enhancement
        sharpened_points = self._edge_sharpening_and_boundary_enhancement(noise_reduced_points, edge_map, normal_map)
        
        # Step 5: Structural detail refinement
        refined_points = self._structural_detail_refinement(sharpened_points, edge_map, curvature_map, normal_map)
        
        # Step 6: Intelligent point redistribution
        redistributed_points = self._intelligent_point_redistribution(refined_points, edge_map, curvature_map, density_map)
        
        # Step 7: Final clarity enhancement
        final_points = self._final_clarity_enhancement(redistributed_points, edge_map, curvature_map)
        
        # Create enhanced point cloud
        enhanced_pc = self._create_point_cloud(final_points, pc.channels)
        
        return enhanced_pc
    
    def _advanced_geometric_analysis(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advanced geometric analysis for edge, curvature, normal, and density detection."""
        try:
            tree = KDTree(points)
            edge_map = np.zeros(len(points))
            curvature_map = np.zeros(len(points))
            normal_map = np.zeros((len(points), 3))
            density_map = np.zeros(len(points))
            
            for i in range(len(points)):
                # Find adaptive neighborhood size
                distances, indices = tree.query(points[i], k=min(30, len(points)))
                neighbors = points[indices]
                
                if len(neighbors) >= 12:
                    # Advanced PCA with adaptive neighborhood
                    centered = neighbors - np.mean(neighbors, axis=0)
                    cov_matrix = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                    
                    # Sort eigenvalues and eigenvectors
                    idx = np.argsort(eigenvalues)
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    
                    if eigenvalues[2] > 1e-8:
                        # Enhanced edge detection
                        planarity = eigenvalues[0] / eigenvalues[2]
                        linearity = eigenvalues[1] / eigenvalues[2]
                        omnivariance = (eigenvalues[0] * eigenvalues[1] * eigenvalues[2]) ** (1/3)
                        anisotropy = (eigenvalues[2] - eigenvalues[0]) / eigenvalues[2]
                        
                        # Comprehensive edge strength
                        edge_strength = (1.0 - planarity) * 0.4 + (1.0 - linearity) * 0.3 + \
                                      (1.0 - omnivariance) * 0.2 + anisotropy * 0.1
                        edge_map[i] = np.clip(edge_strength, 0, 1)
                        
                        # Enhanced curvature estimation
                        curvature_map[i] = eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2])
                        
                        # Normal vector (smallest eigenvalue direction)
                        normal_map[i] = eigenvectors[:, 0]
                        
                        # Local density estimation
                        local_density = len(neighbors) / (np.mean(distances) + 1e-8)
                        density_map[i] = local_density
                    else:
                        edge_map[i] = 0.0
                        curvature_map[i] = 0.0
                        normal_map[i] = np.array([0, 0, 1])
                        density_map[i] = 0.0
                else:
                    edge_map[i] = 0.0
                    curvature_map[i] = 0.0
                    normal_map[i] = np.array([0, 0, 1])
                    density_map[i] = 0.0
            
            # Normalize maps
            edge_map = np.clip(edge_map, 0, 1)
            curvature_map = np.clip(curvature_map, 0, 1)
            density_map = np.clip(density_map / np.max(density_map + 1e-8), 0, 1)
            
            return edge_map, curvature_map, normal_map, density_map
            
        except Exception as e:
            self.logger.warning(f"Advanced geometric analysis failed: {e}")
            return np.zeros(len(points)), np.zeros(len(points)), np.zeros((len(points), 3)), np.zeros(len(points))
    
    def _intelligent_noise_reduction(self, points: np.ndarray, edge_map: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """Intelligent noise reduction that preserves edges and important features."""
        try:
            tree = KDTree(points)
            denoised = points.copy()
            
            for i in range(len(points)):
                edge_strength = edge_map[i]
                local_density = density_map[i]
                
                # Adaptive filtering based on edge strength and density
                if edge_strength > 0.7:  # Strong edge - minimal filtering
                    filter_strength = 0.05
                    neighborhood_size = 8
                elif edge_strength > 0.4:  # Medium edge - light filtering
                    filter_strength = 0.12
                    neighborhood_size = 12
                elif edge_strength > 0.2:  # Weak edge - moderate filtering
                    filter_strength = 0.20
                    neighborhood_size = 16
                else:  # Flat area - more filtering
                    filter_strength = 0.30
                    neighborhood_size = 20
                
                # Adjust based on local density
                if local_density > 0.7:  # High density - more filtering
                    filter_strength *= 1.2
                elif local_density < 0.3:  # Low density - less filtering
                    filter_strength *= 0.8
                
                # Apply intelligent filtering
                distances, indices = tree.query(points[i], k=min(neighborhood_size, len(points)))
                neighbors = points[indices]
                
                # Edge-aware weights
                spatial_weights = np.exp(-distances**2 / (2 * 0.01**2))
                
                # Consider edge strength of neighbors
                edge_weights = 1.0 - edge_map[indices] * 0.5  # Reduce weight of edge neighbors
                
                # Combined weights
                combined_weights = spatial_weights * edge_weights
                combined_weights /= np.sum(combined_weights)
                
                # Apply filtering
                local_mean = np.sum(neighbors * combined_weights[:, np.newaxis], axis=0)
                denoised[i] = points[i] * (1 - filter_strength) + local_mean * filter_strength
            
            return denoised
            
        except Exception as e:
            self.logger.warning(f"Intelligent noise reduction failed: {e}")
            return points
    
    def _edge_sharpening_and_boundary_enhancement(self, points: np.ndarray, edge_map: np.ndarray, normal_map: np.ndarray) -> np.ndarray:
        """Enhance edge sharpness and boundary definition."""
        try:
            tree = KDTree(points)
            sharpened = points.copy()
            
            for i in range(len(points)):
                edge_strength = edge_map[i]
                normal = normal_map[i]
                
                if edge_strength > 0.3:  # Only process edge regions
                    # Find neighbors for local analysis
                    distances, indices = tree.query(points[i], k=min(15, len(points)))
                    neighbors = points[indices]
                    
                    # Adaptive sharpening based on edge strength
                    if edge_strength > 0.8:  # Very strong edge
                        sharpen_factor = 1.3
                        projection_strength = 0.8
                    elif edge_strength > 0.6:  # Strong edge
                        sharpen_factor = 1.2
                        projection_strength = 0.6
                    elif edge_strength > 0.4:  # Medium edge
                        sharpen_factor = 1.1
                        projection_strength = 0.4
                    else:  # Weak edge
                        sharpen_factor = 1.05
                        projection_strength = 0.2
                    
                    # Local surface analysis
                    if len(neighbors) >= 6:
                        # Fit local plane
                        local_center = np.mean(neighbors, axis=0)
                        local_centered = neighbors - local_center
                        local_cov = np.cov(local_centered.T)
                        local_eigenvalues, local_eigenvectors = np.linalg.eigh(local_cov)
                        
                        # Project point onto local surface
                        point_to_center = points[i] - local_center
                        projection = np.dot(point_to_center, local_eigenvectors[:, 0]) * local_eigenvectors[:, 0]
                        surface_point = points[i] - projection
                        
                        # Apply edge sharpening
                        sharpened_direction = points[i] - surface_point
                        if np.linalg.norm(sharpened_direction) > 1e-8:
                            sharpened_direction /= np.linalg.norm(sharpened_direction)
                            sharpening_offset = sharpened_direction * (sharpen_factor - 1.0) * 0.015
                            sharpened[i] = surface_point + (points[i] - surface_point) * sharpen_factor + \
                                         sharpening_offset * projection_strength
            
            return sharpened
            
        except Exception as e:
            self.logger.warning(f"Edge sharpening and boundary enhancement failed: {e}")
            return points
    
    def _structural_detail_refinement(self, points: np.ndarray, edge_map: np.ndarray, curvature_map: np.ndarray, normal_map: np.ndarray) -> np.ndarray:
        """Refine structural details while preserving important features."""
        try:
            tree = KDTree(points)
            refined = points.copy()
            
            for i in range(len(points)):
                edge_strength = edge_map[i]
                curvature = curvature_map[i]
                normal = normal_map[i]
                
                # Different refinement strategies based on local geometry
                if edge_strength > 0.6:  # Edge region
                    # Edge preservation with subtle refinement
                    distances, indices = tree.query(points[i], k=min(10, len(points)))
                    neighbors = points[indices]
                    
                    # Edge-preserving refinement
                    weights = np.exp(-distances**2 / (2 * 0.006**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    refined[i] = points[i] * 0.92 + local_mean * 0.08
                    
                elif curvature > 0.4:  # High curvature region
                    # Detail enhancement in curved areas
                    distances, indices = tree.query(points[i], k=min(18, len(points)))
                    neighbors = points[indices]
                    
                    # Curvature-aware refinement
                    weights = np.exp(-distances**2 / (2 * 0.008**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    local_var = np.var(neighbors, axis=0)
                    
                    # Enhance based on curvature
                    detail_factor = 1.0 + curvature * 0.25
                    refined[i] = local_mean + (points[i] - local_mean) * detail_factor
                    
                else:  # Flat or low-detail region
                    # Surface quality improvement
                    distances, indices = tree.query(points[i], k=min(25, len(points)))
                    neighbors = points[indices]
                    
                    # Quality enhancement for flat areas
                    weights = np.exp(-distances**2 / (2 * 0.012**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    refined[i] = points[i] * 0.85 + local_mean * 0.15
            
            return refined
            
        except Exception as e:
            self.logger.warning(f"Structural detail refinement failed: {e}")
            return points
    
    def _intelligent_point_redistribution(self, points: np.ndarray, edge_map: np.ndarray, curvature_map: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """Intelligently redistribute points for better coverage and detail."""
        try:
            tree = KDTree(points)
            
            # Calculate redistribution priority
            priority_map = edge_map * 0.5 + curvature_map * 0.3 + (1.0 - density_map) * 0.2
            
            # Find regions that need more points
            redistribution_candidates = []
            target_addition = int(len(points) * 0.12)  # 12% increase for better coverage
            
            # Sort by priority
            priority_indices = np.argsort(priority_map)[::-1]
            
            for idx in priority_indices:
                if len(redistribution_candidates) >= target_addition:
                    break
                
                if priority_map[idx] > 0.25:  # Only in important regions
                    base_point = points[idx]
                    
                    # Check local density
                    distances, _ = tree.query(base_point, k=min(10, len(points)))
                    local_density = 1.0 / (np.mean(distances) + 1e-8)
                    
                    # Add points where density is low but importance is high
                    if local_density < 120 and priority_map[idx] > 0.4:
                        # Number of points to add based on priority
                        points_to_add = min(3, int(priority_map[idx] * 4))
                        
                        for _ in range(points_to_add):
                            if len(redistribution_candidates) >= target_addition:
                                break
                            
                            # Create new point with geometric awareness
                            if edge_map[idx] > 0.5:
                                # Edge-following point placement
                                # Find edge direction
                                distances, indices = tree.query(base_point, k=min(8, len(points)))
                                neighbors = points[indices]
                                
                                if len(neighbors) >= 4:
                                    centered = neighbors - np.mean(neighbors, axis=0)
                                    cov_matrix = np.cov(centered.T)
                                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                                    
                                    # Follow edge direction
                                    edge_direction = eigenvectors[:, 2]
                                    offset = edge_direction * np.random.uniform(-0.008, 0.008)
                            else:
                                # Curved surface point placement
                                offset = np.random.randn(3) * 0.010
                            
                            new_point = base_point + offset
                            
                            # Ensure minimum distance
                            distances_check, _ = tree.query(new_point, k=1)
                            if distances_check > 0.003:
                                redistribution_candidates.append(new_point)
            
            if redistribution_candidates:
                # Combine original and redistributed points
                combined = np.vstack([points, np.array(redistribution_candidates)])
                return combined
            
            return points
            
        except Exception as e:
            self.logger.warning(f"Intelligent point redistribution failed: {e}")
            return points
    
    def _final_clarity_enhancement(self, points: np.ndarray, edge_map: np.ndarray, curvature_map: np.ndarray) -> np.ndarray:
        """Final clarity enhancement for overall quality improvement."""
        try:
            tree = KDTree(points)
            final = points.copy()
            
            for i in range(len(points)):
                edge_strength = edge_map[i]
                curvature = curvature_map[i]
                
                # Final enhancement based on local geometry
                if edge_strength > 0.5:  # Edge region
                    # Final edge enhancement
                    distances, indices = tree.query(points[i], k=min(12, len(points)))
                    neighbors = points[indices]
                    
                    # Edge-preserving enhancement
                    weights = np.exp(-distances**2 / (2 * 0.005**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    
                    # Final edge sharpening
                    edge_factor = 1.0 + edge_strength * 0.15
                    final[i] = local_mean + (points[i] - local_mean) * edge_factor
                    
                elif curvature > 0.3:  # High curvature
                    # Final detail enhancement
                    distances, indices = tree.query(points[i], k=min(16, len(points)))
                    neighbors = points[indices]
                    
                    # Curvature-aware enhancement
                    weights = np.exp(-distances**2 / (2 * 0.007**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    
                    # Final curvature enhancement
                    detail_factor = 1.0 + curvature * 0.2
                    final[i] = local_mean + (points[i] - local_mean) * detail_factor
                    
                else:  # Flat area
                    # Final surface quality enhancement
                    distances, indices = tree.query(points[i], k=min(20, len(points)))
                    neighbors = points[indices]
                    
                    # Quality enhancement
                    weights = np.exp(-distances**2 / (2 * 0.01**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    final[i] = points[i] * 0.88 + local_mean * 0.12
            
            return final
            
        except Exception as e:
            self.logger.warning(f"Final clarity enhancement failed: {e}")
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
