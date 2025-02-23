import numpy as np
from point_e.enhancements.normalizepointcloud import normalize_pointcloud
from point_e.enhancements.bilateralsmoothing import bilateral_smoothing
from point_e.util.point_cloud import PointCloud

def enhance_point_cloud(pc):
    """
    Apply the full enhancement pipeline to a Point-E point cloud object.
    
    Args:
        pc (PointCloud): Point-E PointCloud object obtained from sampler.
        
    Returns:
        PointCloud: Enhanced Point-E PointCloud object for visualization.
    """
    # Extract coordinates from Point-E PointCloud
    points = np.array(pc.coords)
    
    # Step 1: Normalize Point Cloud
    normalized_points = normalize_pointcloud(points)
    
    # Step 2: Apply Bilateral Smoothing
    smoothed_points = bilateral_smoothing(normalized_points)
    
    # Rebuild Point-E PointCloud Object for Compatibility
    enhanced_pc = PointCloud(coords=smoothed_points, channels=pc.channels)
    
    return enhanced_pc
