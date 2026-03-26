import numpy as np
from point_e.enhancements.normalizepointcloud import normalize_pointcloud
from point_e.enhancements.bilateralsmoothing import bilateral_smoothing
from point_e.enhancements.production_enhancement import enhance_point_cloud_production
from point_e.util.point_cloud import PointCloud

def enhance_point_cloud(pc, production_mode=True, prompt="", save_ply=True, optimize_density=True):
    """
    Apply the full enhancement pipeline to a Point-E point cloud object.
    
    Args:
        pc (PointCloud): Point-E PointCloud object obtained from sampler.
        production_mode (bool): Use production-ready enhancement (default: True)
        prompt (str): Text prompt used for generation (for metadata)
        save_ply (bool): Whether to save as PLY file (production mode only)
        optimize_density (bool): Whether to optimize point density (production mode only)
        
    Returns:
        PointCloud: Enhanced Point-E PointCloud object for visualization.
    """
    if production_mode:
        # Use production-ready enhancement
        return enhance_point_cloud_production(pc, prompt, save_ply, optimize_density)
    else:
        # Legacy enhancement pipeline
        # Extract coordinates from Point-E PointCloud
        points = np.array(pc.coords)
        
        # Step 1: Normalize Point Cloud
        normalized_points = normalize_pointcloud(points)
        
        # Step 2: Apply Bilateral Smoothing
        smoothed_points = bilateral_smoothing(normalized_points)
        
        # Rebuild Point-E PointCloud Object for Compatibility
        enhanced_pc = PointCloud(coords=smoothed_points, channels=pc.channels)
        
        return enhanced_pc
