"""
Advanced point cloud enhancements using Open3D.
Includes density enhancement, structural accuracy improvement, and surface reconstruction.
"""

import numpy as np
import logging
from typing import Tuple, Optional

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    o3d = None
    OPEN3D_AVAILABLE = False

from .bilateralsmoothing import bilateral_smoothing
from .normalizepointcloud import normalize_pointcloud
from point_e.util.point_cloud import PointCloud

logger = logging.getLogger(__name__)


def densify_point_cloud(
    points: np.ndarray,
    target_point_count: int,
    method: str = "interpolation",
) -> np.ndarray:
    """
    Increase point cloud density using interpolation or mesh reconstruction.
    
    Methods:
    - "interpolation": KDTree-based interpolation (fast + shape preserving)
    - "upsample": Simple replication with noise (fast, minimal shape change)
    - "poisson": Surface reconstruction + mesh sampling (aggressive; avoided by default)
    
    Args:
        points: Original point cloud [N, 3]
        target_point_count: Desired number of points
        method: Densification method
    
    Returns:
        Densified point cloud [M, 3] where M ≥ target_point_count
    """
    current_count = len(points)
    if current_count >= target_point_count:
        # Downsample if needed
        indices = np.random.choice(current_count, target_point_count, replace=False)
        return points[indices]

    if method == "poisson":
        if not OPEN3D_AVAILABLE:
            logging.warning("Open3D not available, falling back to interpolation densification.")
            method = "interpolation"
        else:
            return _densify_poisson(points, target_point_count)

    if method == "interpolation":
        return _densify_interpolation(points, target_point_count)
    elif method == "upsample":
        return _densify_upsample(points, target_point_count)
    else:
        raise ValueError(f"Unknown densification method: {method}")


def _densify_poisson(points: np.ndarray, target_count: int) -> np.ndarray:
    """
    Densify via Poisson surface reconstruction.
    Most accurate but slowest method.
    """
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D unavailable for Poisson densification. Falling back to interpolation.")
        return _densify_interpolation(points, target_count)

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=9,
            width=0,
            scale=1.1,
            linear_fit=False,
        )
        if mesh is None or len(mesh.vertices) == 0:
            raise RuntimeError("Poisson reconstruction produced empty mesh")

        # Sample points from mesh
        densified = mesh.sample_points_uniformly(number_of_points=target_count)
        if len(densified.points) == 0:
            raise RuntimeError("Mesh sampling returned zero points")

        return np.asarray(densified.points)
    except Exception as e:
        logger.warning(f"Poisson reconstruction failed: {e}, falling back to interpolation")
        return _densify_interpolation(points, target_count)


def _densify_interpolation(points: np.ndarray, target_count: int) -> np.ndarray:
    """
    Densify via KDTree interpolation.
    Fast and reliable, adds points in undersampled regions.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    densified = [points.copy()]
    current_count = len(points)
    
    while current_count < target_count:
        # Sample random regions
        num_new = min(len(points), target_count - current_count)
        base_indices = np.random.choice(len(points), num_new)
        
        new_points = []
        for idx in base_indices:
            # Find K nearest neighbors
            _, neighbor_indices, _ = kdtree.search_knn_vector_3d(points[idx], 5)
            
            # Interpolate between this point and a random neighbor
            neighbor_idx = neighbor_indices[np.random.randint(1, len(neighbor_indices))]
            alpha = np.random.rand()
            new_point = (1 - alpha) * points[idx] + alpha * points[neighbor_idx]
            new_points.append(new_point)
        
        densified.append(np.array(new_points))
        current_count += len(new_points)
    
    result = np.vstack(densified)
    indices = np.random.choice(len(result), target_count, replace=False)
    return result[indices]


def _densify_upsample(points: np.ndarray, target_count: int) -> np.ndarray:
    """
    Densify via simple replication with noise.
    Fastest but lowest quality.
    """
    ratio = target_count / len(points)
    num_replications = int(np.ceil(ratio))
    
    densified = []
    for _ in range(num_replications):
        noise = np.random.randn(*points.shape) * 0.01
        densified.append(points + noise)
    
    result = np.vstack(densified)
    indices = np.random.choice(len(result), target_count, replace=False)
    return result[indices]


def improve_structural_accuracy(
    points: np.ndarray,
    smoothing_iterations: int = 2,
    outlier_removal: bool = True,
) -> np.ndarray:
    """
    Improve structural accuracy of point cloud while preserving shape.
    
    Applies:
    1. Outlier removal (statistical)
    2. Edge-preserving bilateral smoothing (light)
    3. Avoids strong surface reconstruction/shape change
    
    Args:
        points: Original point cloud [N, 3]
        smoothing_iterations: Number of smoothing passes
        outlier_removal: Whether to remove outliers
    
    Returns:
        Structurally improved point cloud
    """
    if OPEN3D_AVAILABLE:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Remove statistical outliers
        if outlier_removal:
            try:
                pcd, indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                logger.info(f"Removed {len(points) - len(pcd.points)} outliers")
                points = np.asarray(pcd.points)
            except Exception as e:
                logger.warning(f"Outlier removal failed: {e} - skipping")
    else:
        logger.warning("Open3D unavailable; skipping outlier removal.")

    # Smooth iteratively to improve structural accuracy
    for i in range(smoothing_iterations):
        points = bilateral_smoothing(
            points,
            spatial_sigma=0.03,
            intensity_sigma=0.02,
            radius=0.12,
            num_iterations=1,
        )
        logger.debug(f"Structural smoothing iteration {i + 1}/{smoothing_iterations}")

    return points


def enhance_point_cloud_advanced(
    pc: PointCloud,
    target_density: int = 8192,
    densification_method: str = "interpolation",
    smooth_iterations: int = 1,
    improve_structure: bool = True,
) -> PointCloud:
    """
    Apply full enhancement pipeline for density and structural accuracy.
    
    Args:
        pc: Input PointCloud object
        target_density: Target number of points after densification
        densification_method: How to increase density ("poisson", "interpolation", "upsample")
        smooth_iterations: Bilateral smoothing passes
        improve_structure: Apply structural accuracy improvements
    
    Returns:
        Enhanced PointCloud object
    """
    points = np.array(pc.coords)
    original_count = len(points)
    
    # Step 1: Normalize to standard space
    points = normalize_pointcloud(points)
    
    # Step 2: Improve structure if requested
    if improve_structure:
        logger.info("Improving structural accuracy...")
        points = improve_structural_accuracy(
            points,
            smoothing_iterations=smooth_iterations,
            outlier_removal=True,
        )
    
    # Step 3: Densify point cloud (use interpolation/upsample by default to preserve shape)
    if target_density > len(points):
        logger.info(f"Densifying from {len(points)} to {target_density} points...")
        method = densification_method
        if method == "poisson":
            logger.warning("Poisson densification requested, but this may alter shape; using interpolation for shape preservation.")
            method = "interpolation"

        points = densify_point_cloud(
            points,
            target_density,
            method=method,
        )
    else:
        logger.info("Target density <= current count; skipping densification.")
    
    # Step 4: Final smoothing pass
    if smooth_iterations > 0:
        logger.info(f"Final smoothing ({smooth_iterations} iterations)...")
        points = bilateral_smoothing(
            points,
            spatial_sigma=0.1,
            intensity_sigma=0.1,
            num_iterations=smooth_iterations,
        )
    
    # Rebuild Point-E PointCloud object with channel transfer
    new_channels = {}
    if all(c in pc.channels for c in ['R', 'G', 'B']):
        if OPEN3D_AVAILABLE:
            # Transfer colors from nearest original points through KD-tree search
            pcd_source = o3d.geometry.PointCloud()
            pcd_source.points = o3d.utility.Vector3dVector(np.asarray(pc.coords))
            kdtree = o3d.geometry.KDTreeFlann(pcd_source)

            color_out = []
            for point in points:
                _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
                nearest = idx[0] if idx else 0
                color_out.append([pc.channels['R'][nearest], pc.channels['G'][nearest], pc.channels['B'][nearest]])

            color_out = np.array(color_out)
            new_channels['R'] = color_out[:, 0]
            new_channels['G'] = color_out[:, 1]
            new_channels['B'] = color_out[:, 2]
        else:
            # Fallback to basic channel interpolation by repeating median color
            mean_color = np.array([np.median(pc.channels['R']), np.median(pc.channels['G']), np.median(pc.channels['B'])])
            new_channels['R'] = np.full(len(points), mean_color[0])
            new_channels['G'] = np.full(len(points), mean_color[1])
            new_channels['B'] = np.full(len(points), mean_color[2])
    else:
        # Keep existing channels if counts match, else fallback to neutral
        for k, v in pc.channels.items():
            if len(v) == len(points):
                new_channels[k] = v
            else:
                new_channels[k] = np.zeros(len(points))

    enhanced_pc = PointCloud(coords=points, channels=new_channels)
    
    logger.info(
        f"Enhancement complete: {original_count} → {len(points)} points "
        f"({100 * (len(points) - original_count) / original_count:.1f}% density increase)"
    )
    
    return enhanced_pc


def compute_point_cloud_metrics(
    points: np.ndarray,
) -> dict:
    """
    Compute quality metrics for a point cloud.
    
    Returns:
        Dictionary with metrics:
        - density: Points per unit volume
        - coverage: Fraction of bounding box with points
        - uniformity: How uniformly distributed points are
        - noise_level: Estimated noise in point positions
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Basic metrics
    bbox = pcd.get_axis_aligned_bounding_box()
    volume = bbox.volume()
    
    # Uniformity: compute average nearest neighbor distance
    pcd_kdtree = o3d.geometry.KDTreeFlann(pcd)
    distances = []
    for i in range(min(100, len(points))):  # Sample for speed
        _, indices, _ = pcd_kdtree.search_knn_vector_3d(points[i], 2)
        if len(indices) > 1:
            distances.append(np.linalg.norm(points[indices[1]] - points[i]))
    
    avg_neighbor_distance = np.mean(distances) if distances else 0
    
    return {
        "num_points": len(points),
        "volume": volume,
        "density_points_per_unit": len(points) / volume if volume > 0 else 0,
        "avg_nearest_neighbor_distance": avg_neighbor_distance,
    }
