"""
Test script for geometric quality enhancements
"""

import numpy as np
from point_e.enhancements.geometric_quality_enhancements import enhance_geometric_quality
from point_e.util.point_cloud import PointCloud

# Create a simple test point cloud (cube)
def create_test_cube():
    # Create a simple cube point cloud
    points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                points.append([x, y, z])

    # Add some noise and extra points
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, (8, 3))
    points = np.array(points) + noise

    # Add some scattered points
    scattered = np.random.uniform(-2, 2, (20, 3))
    points = np.vstack([points, scattered])

    return points

# Test the enhancement
print("Testing geometric quality enhancements...")

# Create test point cloud
points = create_test_cube()
print(f"Original points: {len(points)}")

# Create PointCloud object
pc = PointCloud(coords=points, channels={})

# Apply enhancement
enhanced_pc = enhance_geometric_quality(pc, target_density_increase=0.3)

print(f"Enhanced points: {len(enhanced_pc.coords)}")
print("Enhancement test completed successfully!")