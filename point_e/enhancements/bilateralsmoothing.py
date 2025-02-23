import numpy as np
import open3d as o3d

def bilateral_smoothing(points, radius=0.1, sigma_color=0.1, sigma_space=0.1):
    """
    Apply bilateral smoothing to a point cloud.
    
    Args:
        points (np.ndarray): Nx3 array representing the point cloud.
        radius (float): Neighborhood radius for smoothing.
        sigma_color (float): Color (intensity) difference sigma for smoothing.
        sigma_space (float): Spatial distance sigma for smoothing.
        
    Returns:
        np.ndarray: Smoothed Nx3 point cloud.
    """
    # Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals for consistent smoothing
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    
    # Convert to numpy array for processing
    points_np = np.asarray(pcd.points)
    normals_np = np.asarray(pcd.normals)
    
    # Apply bilateral smoothing
    smoothed_points = np.zeros_like(points_np)
    for i in range(len(points_np)):
        # Get neighbors within the radius
        distances = np.linalg.norm(points_np - points_np[i], axis=1)
        neighbors = points_np[distances < radius]
        
        # Bilateral weighting
        spatial_weights = np.exp(-0.5 * (distances[distances < radius] / sigma_space) ** 2)
        color_weights = np.exp(-0.5 * (np.linalg.norm(neighbors - points_np[i], axis=1) / sigma_color) ** 2)
        weights = spatial_weights * color_weights
        
        # Normalize weights and smooth the point
        weights /= np.sum(weights)
        smoothed_points[i] = np.sum(neighbors * weights[:, np.newaxis], axis=0)
    
    return smoothed_points
