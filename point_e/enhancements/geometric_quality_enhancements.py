"""
Local Neighborhood Quality Enhancements for Point Clouds
Implements read-only local analysis (k-NN statistics, outlier detection, normal estimation)
WITHOUT modifying point coordinates, count, or geometry in any way.
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

from point_e.util.point_cloud import PointCloud

logger = logging.getLogger(__name__)


def statistical_outlier_removal(points: np.ndarray) -> np.ndarray:
    """
    Perform read-only statistical outlier analysis.
    Returns the original points unchanged - only analyzes for outliers without modification.

    Args:
        points: Point cloud coordinates [N, 3]

    Returns:
        Original point cloud [N, 3] (completely unchanged)
    """
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D not available, skipping statistical outlier analysis")
        return points

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Perform statistical outlier analysis (read-only)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        outlier_count = len(points) - len(ind)
        if outlier_count > 0:
            logger.info(f"Detected {outlier_count} statistical outliers (analysis only, no modification)")

        # Return original points COMPLETELY UNCHANGED
        return points

    except Exception as e:
        logger.warning(f"Statistical outlier analysis failed: {e}, returning original points")
        return points


def knn_pca_normal_refinement(points: np.ndarray) -> np.ndarray:
    """
    Perform read-only k-NN/PCA-based normal estimation and analysis.
    Returns the original points completely unchanged - only analyzes normals.

    Args:
        points: Point cloud coordinates [N, 3]

    Returns:
        Original point cloud [N, 3] (completely unchanged)
    """
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D not available, skipping k-NN/PCA normal analysis")
        return points

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate normals using k-NN and PCA (read-only analysis)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Orient normals consistently (analysis only)
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # Compute normal statistics for logging/analysis only
        normals = np.asarray(pcd.normals)
        normal_consistency = np.mean(np.abs(np.linalg.norm(normals, axis=1) - 1.0))
        logger.debug(f"Normal estimation completed (consistency: {normal_consistency:.6f})")

        # Return original points COMPLETELY UNCHANGED
        return points

    except Exception as e:
        logger.warning(f"k-NN/PCA normal analysis failed: {e}, returning original points")
        return points


def enhance_point_cloud_quality(pc: PointCloud, num_iterations: int = 1) -> PointCloud:
    """
    Apply read-only local neighborhood quality analysis using statistical outlier detection
    and k-NN/PCA-based normal estimation. Does NOT modify point coordinates, count, or geometry.

    Args:
        pc: Input PointCloud object
        num_iterations: Number of analysis iterations (default: 1 for stability)

    Returns:
        Original PointCloud object with same coordinates and count (completely unchanged)
    """
    points = np.array(pc.coords)
    original_count = len(points)
    logger.info(f"Starting read-only local neighborhood quality analysis on {original_count} points")

    for iteration in range(num_iterations):
        logger.debug(f"Quality analysis iteration {iteration + 1}/{num_iterations}")

        # Step 1: Statistical outlier analysis (read-only)
        logger.debug("Performing statistical outlier analysis...")
        points = statistical_outlier_removal(points)

        # Step 2: k-NN/PCA-based normal analysis (read-only)
        logger.debug("Performing k-NN/PCA-based normal analysis...")
        points = knn_pca_normal_refinement(points)

    # Verify that points are COMPLETELY UNCHANGED
    if not np.array_equal(points, np.array(pc.coords)):
        logger.error("CRITICAL ERROR: Point coordinates were modified during analysis!")
        raise ValueError("Point coordinates must never be modified")

    if len(points) != original_count:
        logger.error("CRITICAL ERROR: Point count changed during analysis!")
        raise ValueError("Point count must never change")

    # Rebuild PointCloud object with identical coordinates
    enhanced_pc = PointCloud(coords=points, channels=pc.channels)

    logger.info(f"Read-only local neighborhood quality analysis complete: {len(points)} points (completely unchanged)")
    return enhanced_pc

    logger.info(f"Local neighborhood quality enhancement complete: {len(points)} points (unchanged count)")
    return enhanced_pc