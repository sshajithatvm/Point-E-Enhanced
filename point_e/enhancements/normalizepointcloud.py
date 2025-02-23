import open3d as o3d
import numpy as np

def normalize_pointcloud(points):
    """
    Normalize a point cloud to fit within a unit sphere.
    
    Args:
        points (np.ndarray): Nx3 array representing the point cloud.
        
    Returns:
        np.ndarray: Normalized Nx3 point cloud.
    """
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Calculate maximum distance from the origin
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    
    # Scale points to fit within the unit sphere
    points = points / max_dist
    
    return points
