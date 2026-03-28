import open3d as o3d
import numpy as np
import os
import uuid
from datetime import datetime
from point_e.util.point_cloud import PointCloud

class ProductionEnhancer:
    """
    Production-ready point cloud enhancement with denoising and density optimization.
    Features automatic PLY export with unique naming and proper folder structure.
    """
    
    def __init__(self, output_dir="outputs/enhanced_outputs"):
        """
        Initialize the production enhancer.
        
        Args:
            output_dir (str): Base directory for enhanced outputs
        """
        self.output_dir = output_dir
        self._ensure_output_structure()
    
    def _ensure_output_structure(self):
        """Create proper folder structure for outputs."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "ply_files"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
    
    def _generate_unique_filename(self, prefix="enhanced_pc", extension=".ply"):
        """Generate unique filename with timestamp and UUID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_{unique_id}{extension}"
    
    def _statistical_outlier_removal(self, pcd, nb_neighbors=20, std_ratio=2.0):
        """Remove statistical outliers from point cloud."""
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return cl, ind
    
    def _radius_outlier_removal(self, pcd, nb_points=16, radius=0.05):
        """Remove radius outliers from point cloud."""
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        return cl, ind
    
    def _voxel_downsampling(self, pcd, voxel_size=0.01):
        """Optimize point cloud density using voxel grid downsampling."""
        return pcd.voxel_down_sample(voxel_size=voxel_size)
    
    def _uniform_downsampling(self, pcd, every_k_points=5):
        """Alternative density optimization using uniform downsampling."""
        return pcd.uniform_down_sample(every_k_points=every_k_points)
    
    def _adaptive_denoising(self, points, noise_factor=0.02):
        """
        Adaptive denoising based on local point density.
        Uses bilateral filtering with custom implementation.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals for better denoising
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )
        
        # Custom bilateral filtering implementation
        points_array = np.asarray(pcd.points)
        normals_array = np.asarray(pcd.normals)
        
        # Parameters for bilateral filtering
        sigma_s = 0.05  # Spatial sigma
        sigma_r = 0.03  # Range sigma
        
        denoised_points = np.zeros_like(points_array)
        
        for i in range(len(points_array)):
            # Find neighbors
            distances = np.linalg.norm(points_array - points_array[i], axis=1)
            neighbors_mask = distances < (3 * sigma_s)  # Adaptive radius
            
            if np.sum(neighbors_mask) < 3:  # Skip if too few neighbors
                denoised_points[i] = points_array[i]
                continue
            
            # Get neighbor points and distances
            neighbor_points = points_array[neighbors_mask]
            neighbor_distances = distances[neighbors_mask]
            
            # Spatial weights
            spatial_weights = np.exp(-0.5 * (neighbor_distances / sigma_s) ** 2)
            
            # Range weights (based on normal similarity)
            if len(normals_array) > i:
                normal_diff = np.linalg.norm(normals_array[neighbors_mask] - normals_array[i], axis=1)
                range_weights = np.exp(-0.5 * (normal_diff / sigma_r) ** 2)
            else:
                range_weights = np.ones_like(spatial_weights)
            
            # Combined weights
            weights = spatial_weights * range_weights
            weights /= np.sum(weights)  # Normalize
            
            # Weighted average
            denoised_points[i] = np.sum(neighbor_points * weights[:, np.newaxis], axis=0)
        
        return denoised_points
    
    def _surface_reconstruction_smoothing(self, points):
        """
        Advanced smoothing using surface reconstruction approach.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # Custom Laplacian smoothing implementation
        points_array = np.asarray(pcd.points)
        normals_array = np.asarray(pcd.normals)
        
        smoothed_points = points_array.copy()
        lambda_filter = 0.5
        iterations = 5
        
        for _ in range(iterations):
            new_points = smoothed_points.copy()
            
            for i in range(len(smoothed_points)):
                # Find neighbors
                distances = np.linalg.norm(smoothed_points - smoothed_points[i], axis=1)
                neighbors_mask = (distances < 0.1) & (distances > 0)  # Exclude self
                
                if np.sum(neighbors_mask) < 3:
                    continue
                
                # Get neighbor points
                neighbor_points = smoothed_points[neighbors_mask]
                
                # Calculate centroid of neighbors
                centroid = np.mean(neighbor_points, axis=0)
                
                # Laplacian smoothing with normal constraint
                laplacian_vector = centroid - smoothed_points[i]
                
                # Project onto tangent plane (perpendicular to normal)
                if i < len(normals_array):
                    normal = normals_array[i]
                    laplacian_tangent = laplacian_vector - np.dot(laplacian_vector, normal) * normal
                else:
                    laplacian_tangent = laplacian_vector
                
                # Apply smoothing
                new_points[i] = smoothed_points[i] + lambda_filter * laplacian_tangent
            
            smoothed_points = new_points
        
        return smoothed_points
    
    def enhance_point_cloud_production(self, pc, prompt="", save_ply=True, optimize_density=True):
        """
        Production-ready point cloud enhancement pipeline.
        
        Args:
            pc (PointCloud): Input Point-E point cloud
            prompt (str): Text prompt used for generation (for metadata)
            save_ply (bool): Whether to save as PLY file
            optimize_density (bool): Whether to optimize point density
            
        Returns:
            PointCloud: Enhanced point cloud with same structure as input
        """
        print("Starting production enhancement pipeline...")
        
        # Extract coordinates
        points = np.array(pc.coords)
        original_count = len(points)
        print(f"Original point count: {original_count}")
        
        # Step 1: Convert to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Step 2: Statistical outlier removal
        print("Removing statistical outliers...")
        pcd, _ = self._statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0)
        
        # Step 3: Radius outlier removal
        print("Removing radius outliers...")
        pcd, _ = self._radius_outlier_removal(pcd, nb_points=16, radius=0.05)
        
        # Step 4: Adaptive denoising
        print("Applying adaptive denoising...")
        points_array = np.asarray(pcd.points)
        denoised_points = self._adaptive_denoising(points_array)
        
        # Step 5: Surface reconstruction smoothing
        print("Applying surface smoothing...")
        smoothed_points = self._surface_reconstruction_smoothing(denoised_points)
        
        # Step 6: Density optimization
        if optimize_density:
            print("Optimizing point density...")
            pcd_final = o3d.geometry.PointCloud()
            pcd_final.points = o3d.utility.Vector3dVector(smoothed_points)
            
            # Adaptive voxel size based on point count
            if len(smoothed_points) > 5000:
                voxel_size = 0.008
            elif len(smoothed_points) > 2000:
                voxel_size = 0.01
            else:
                voxel_size = 0.015
            
            pcd_final = self._voxel_downsampling(pcd_final, voxel_size=voxel_size)
            final_points = np.asarray(pcd_final.points)
        else:
            final_points = smoothed_points
        
        # Step 7: Create enhanced PointCloud object
        # Handle channels properly - match to number of points
        enhanced_channels = {}
        if pc.channels:
            for channel_name, channel_values in pc.channels.items():
                if len(channel_values) == original_count:
                    # Downsample channels to match final point count
                    if len(final_points) < original_count:
                        # Sample indices for downsampling
                        indices = np.random.choice(original_count, len(final_points), replace=False)
                        enhanced_channels[channel_name] = channel_values[indices]
                    elif len(final_points) == original_count:
                        enhanced_channels[channel_name] = channel_values
                    else:
                        # Upsample if needed (rare case)
                        enhanced_channels[channel_name] = np.interp(
                            np.linspace(0, 1, len(final_points)),
                            np.linspace(0, 1, original_count),
                            channel_values
                        )
                else:
                    # Channel size doesn't match, create default values
                    enhanced_channels[channel_name] = np.ones(len(final_points)) * 0.5
        
        enhanced_pc = PointCloud(coords=final_points, channels=enhanced_channels)
        
        # Step 8: Save PLY file if requested
        if save_ply:
            self._save_ply_file(final_points, prompt, original_count, len(final_points))
        
        print(f"Enhancement complete. Final point count: {len(final_points)}")
        print(f"Point reduction: {((original_count - len(final_points)) / original_count * 100):.1f}%")
        
        return enhanced_pc
    
    def _save_ply_file(self, points, prompt, original_count, final_count):
        """Save enhanced point cloud as PLY file with metadata."""
        # Generate unique filename
        filename = self._generate_unique_filename("enhanced_pointcloud", ".ply")
        ply_path = os.path.join(self.output_dir, "ply_files", filename)
        
        # Create Open3D point cloud and save
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors if available (normalized to [0,1])
        if len(points) > 0:
            # Generate colors based on height for visualization
            heights = points[:, 2]  # Z coordinates
            heights_normalized = (heights - heights.min()) / (heights.max() - heights.min())
            colors = np.zeros((len(points), 3))
            colors[:, 0] = heights_normalized  # Red gradient based on height
            colors[:, 1] = 1.0 - heights_normalized  # Green gradient
            colors[:, 2] = 0.5  # Blue constant
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save PLY file
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"PLY file saved: {ply_path}")
        
        # Save metadata
        metadata = {
            "filename": filename,
            "prompt": prompt,
            "original_point_count": original_count,
            "final_point_count": final_count,
            "enhancement_timestamp": datetime.now().isoformat(),
            "point_reduction_percentage": ((original_count - final_count) / original_count * 100)
        }
        
        metadata_filename = filename.replace(".ply", "_metadata.json")
        metadata_path = os.path.join(self.output_dir, "metadata", metadata_filename)
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved: {metadata_path}")

# Global instance for easy access
_production_enhancer = None

def get_production_enhancer():
    """Get or create the global production enhancer instance."""
    global _production_enhancer
    if _production_enhancer is None:
        _production_enhancer = ProductionEnhancer()
    return _production_enhancer

def enhance_point_cloud_production(pc, prompt="", save_ply=True, optimize_density=True):
    """
    Convenience function for production enhancement.
    
    Args:
        pc (PointCloud): Input Point-E point cloud
        prompt (str): Text prompt used for generation
        save_ply (bool): Whether to save as PLY file
        optimize_density (bool): Whether to optimize point density
        
    Returns:
        PointCloud: Enhanced point cloud
    """
    enhancer = get_production_enhancer()
    return enhancer.enhance_point_cloud_production(pc, prompt, save_ply, optimize_density)
