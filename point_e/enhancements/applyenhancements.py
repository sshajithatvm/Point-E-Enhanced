import numpy as np
from point_e.enhancements.normalizepointcloud import normalize_pointcloud
from point_e.enhancements.geometric_quality_enhancements import enhance_point_cloud_quality
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
    
    # Rebuild PointCloud for geometric enhancement
    normalized_pc = PointCloud(coords=normalized_points, channels=pc.channels)
    
    # Step 2: Apply local neighborhood quality enhancements
    enhanced_pc = enhance_point_cloud_quality(normalized_pc, num_iterations=1)
    
    return enhanced_pc
