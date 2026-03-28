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

class SharpDetailProcessor:
    """Sharp detail processor that preserves edges and enhances natural detail."""
    
    def __init__(self, num_workers: int = mp.cpu_count()):
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self.validator = PointCloudValidator()
    
    def enhance_batch(self, point_clouds: List[object]) -> List[object]:
        """Enhance multiple point clouds with sharp detail preservation."""
        # Validate input point clouds first
        for i, pc in enumerate(point_clouds):
            validation_result = self.validator.validate_point_cloud(pc, min_points=50)
            if not validation_result.is_valid:
                self.logger.error(f"Input point cloud {i} failed validation: {validation_result.errors}")
                continue
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._enhance_sharp_detail, pc) for pc in point_clouds]
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
    
    def _enhance_sharp_detail(self, pc) -> object:
        """Enhance point cloud with sharp detail preservation and intelligent upsampling."""
        points = np.array(pc.coords)
        original_count = len(points)
        
        # Pre-validation
        if len(points) < 50:
            self.logger.warning(f"Point cloud too small for enhancement: {len(points)} points")
            return pc
        
        # Step 1: Preserve all original points
        preserved_points = points.copy()
        
        # Step 2: Edge detection and feature analysis
        edge_map, curvature_map = self._detect_edges_and_curvature(preserved_points)
        
        # Step 3: Selective noise reduction (preserve edges)
        noise_reduced_points = self._selective_noise_reduction(preserved_points, edge_map)
        
        # Step 4: Edge-aware detail enhancement
        enhanced_points = self._edge_aware_detail_enhancement(noise_reduced_points, edge_map, curvature_map)
        
        # Step 5: Intelligent gap filling (focus on edges and curves)
        gap_filled_points = self._intelligent_gap_filling(enhanced_points, edge_map, curvature_map, original_count)
        
        # Step 6: Final edge preservation
        final_points = self._final_edge_preservation(gap_filled_points, edge_map)
        
        # Create enhanced point cloud
        enhanced_pc = self._create_point_cloud(final_points, pc.channels)
        
        return enhanced_pc
    
    def _detect_edges_and_curvature(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect edges and curvature for intelligent processing."""
        try:
            tree = KDTree(points)
            edge_map = np.zeros(len(points))
            curvature_map = np.zeros(len(points))
            
            for i in range(len(points)):
                # Find local neighborhood
                distances, indices = tree.query(points[i], k=min(15, len(points)))
                neighbors = points[indices]
                
                if len(neighbors) >= 6:
                    # Estimate local plane using PCA
                    centered = neighbors - np.mean(neighbors, axis=0)
                    cov_matrix = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                    
                    # Edge strength based on eigenvalue ratios
                    if eigenvalues[2] > 1e-8:  # Avoid division by zero
                        planarity = eigenvalues[0] / eigenvalues[2]
                        linearity = eigenvalues[1] / eigenvalues[2]
                        
                        # Edge strength: high where planarity is low (non-planar)
                        edge_map[i] = 1.0 - planarity
                        
                        # Curvature: based on eigenvalue spread
                        curvature_map[i] = eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2])
                    else:
                        edge_map[i] = 0.0
                        curvature_map[i] = 0.0
                else:
                    edge_map[i] = 0.0
                    curvature_map[i] = 0.0
            
            # Normalize maps
            edge_map = np.clip(edge_map, 0, 1)
            curvature_map = np.clip(curvature_map, 0, 1)
            
            return edge_map, curvature_map
            
        except Exception as e:
            self.logger.warning(f"Edge detection failed: {e}")
            return np.zeros(len(points)), np.zeros(len(points))
    
    def _selective_noise_reduction(self, points: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
        """Apply selective noise reduction that preserves edges."""
        try:
            tree = KDTree(points)
            denoised = np.zeros_like(points)
            
            for i in range(len(points)):
                # Find neighbors
                distances, indices = tree.query(points[i], k=min(12, len(points)))
                neighbors = points[indices]
                
                # Adaptive filtering based on edge strength
                edge_strength = edge_map[i]
                
                if edge_strength > 0.7:  # Strong edge - minimal filtering
                    filter_strength = 0.02
                elif edge_strength > 0.4:  # Medium edge - light filtering
                    filter_strength = 0.08
                elif edge_strength > 0.2:  # Weak edge - moderate filtering
                    filter_strength = 0.15
                else:  # Flat area - more filtering
                    filter_strength = 0.25
                
                # Apply weighted averaging
                weights = np.exp(-distances**2 / (2 * 0.01**2))
                weights /= weights.sum()
                
                local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                denoised[i] = points[i] * (1 - filter_strength) + local_mean * filter_strength
            
            return denoised
            
        except Exception as e:
            self.logger.warning(f"Selective noise reduction failed: {e}")
            return points
    
    def _edge_aware_detail_enhancement(self, points: np.ndarray, edge_map: np.ndarray, curvature_map: np.ndarray) -> np.ndarray:
        """Enhance details while preserving edges."""
        try:
            tree = KDTree(points)
            enhanced = points.copy()
            
            for i in range(len(points)):
                edge_strength = edge_map[i]
                curvature = curvature_map[i]
                
                # Different enhancement strategies based on local geometry
                if edge_strength > 0.6:  # Edge region - preserve sharpness
                    # Minimal enhancement, just slight sharpening
                    distances, indices = tree.query(points[i], k=min(8, len(points)))
                    neighbors = points[indices]
                    
                    # Edge-preserving sharpening
                    weights = np.exp(-distances**2 / (2 * 0.005**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    enhanced[i] = points[i] * 1.1 - local_mean * 0.1  # Sharpening
                    
                elif curvature > 0.5:  # High curvature - enhance detail
                    # Add subtle detail enhancement
                    distances, indices = tree.query(points[i], k=min(10, len(points)))
                    neighbors = points[indices]
                    
                    # Curvature-aware enhancement
                    weights = np.exp(-distances**2 / (2 * 0.008**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    local_var = np.var(neighbors, axis=0)
                    
                    # Enhance based on local variation
                    enhancement_factor = 1.0 + curvature * 0.1
                    enhanced[i] = local_mean + (points[i] - local_mean) * enhancement_factor
                    
                else:  # Flat area - gentle smoothing
                    distances, indices = tree.query(points[i], k=min(15, len(points)))
                    neighbors = points[indices]
                    
                    # Gentle smoothing for flat areas
                    weights = np.exp(-distances**2 / (2 * 0.015**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    enhanced[i] = points[i] * 0.9 + local_mean * 0.1
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Edge-aware detail enhancement failed: {e}")
            return points
    
    def _intelligent_gap_filling(self, points: np.ndarray, edge_map: np.ndarray, curvature_map: np.ndarray, original_count: int) -> np.ndarray:
        """Intelligently fill gaps focusing on edges and curved regions."""
        current_count = len(points)
        
        # Calculate target increase (15-20% for natural enhancement)
        target_increase = int(original_count * 0.18)  # 18% increase
        target_count = original_count + target_increase
        
        if current_count >= target_count:
            return points
        
        # Create priority map for point insertion
        priority_map = edge_map * 0.6 + curvature_map * 0.4  # Prioritize edges and curves
        
        # Find high-priority regions for point insertion
        tree = KDTree(points)
        insertion_candidates = []
        
        for i in range(len(points)):
            if priority_map[i] > 0.3:  # Only in interesting regions
                if len(insertion_candidates) >= target_increase:
                    break
                    
                base_point = points[i]
                
                # Check local density
                distances, _ = tree.query(base_point, k=min(8, len(points)))
                local_density = 1.0 / (np.mean(distances) + 1e-8)
                
                # Insert more points where density is low but priority is high
                if local_density < 100 and priority_map[i] > 0.5:
                    # Number of points to add based on priority and density
                    points_to_add = min(3, int(priority_map[i] * 5))
                    
                    for _ in range(points_to_add):
                        if len(insertion_candidates) >= target_increase:
                            break
                        
                        # Create new point following local geometry
                        # Small random offset with geometric constraints
                        offset_range = 0.008 * (2.0 - priority_map[i])  # Smaller offset for high priority
                        offset = np.random.randn(3) * offset_range
                        
                        # Bias offset towards edge direction
                        if edge_map[i] > 0.5:
                            # Find edge direction
                            distances, indices = tree.query(base_point, k=min(6, len(points)))
                            neighbors = points[indices]
                            
                            if len(neighbors) >= 3:
                                # Estimate edge direction using PCA
                                centered = neighbors - np.mean(neighbors, axis=0)
                                cov_matrix = np.cov(centered.T)
                                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                                
                                # Use the direction with largest eigenvalue (edge direction)
                                edge_direction = eigenvectors[:, 2]
                                offset = offset * 0.3 + np.dot(offset, edge_direction) * edge_direction * 0.7
                        
                        new_point = base_point + offset
                        
                        # Ensure minimum distance from existing points
                        distances_check, _ = tree.query(new_point, k=1)
                        if distances_check > 0.003:  # Minimum distance
                            insertion_candidates.append(new_point)
        
        if insertion_candidates:
            # Combine original and new points
            combined = np.vstack([points, np.array(insertion_candidates)])
            return combined
        
        return points
    
    def _final_edge_preservation(self, points: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
        """Final pass to preserve edge sharpness."""
        try:
            # Apply very light edge-preserving filter
            tree = KDTree(points)
            final = points.copy()
            
            for i in range(len(points)):
                if edge_map[i] > 0.5:  # Edge point
                    # Minimal processing for edge points
                    distances, indices = tree.query(points[i], k=min(5, len(points)))
                    neighbors = points[indices]
                    
                    # Very light averaging to preserve sharpness
                    weights = np.exp(-distances**2 / (2 * 0.003**2))
                    weights /= weights.sum()
                    
                    local_mean = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                    final[i] = points[i] * 0.95 + local_mean * 0.05  # Very light smoothing
                else:
                    # Normal processing for non-edge points
                    final[i] = points[i]  # Keep as is
            
            return final
            
        except Exception as e:
            self.logger.warning(f"Final edge preservation failed: {e}")
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
