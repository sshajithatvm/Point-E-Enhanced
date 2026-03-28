#!/usr/bin/env python
"""
Quick Enhancement Demo (synthetic point cloud) to verify pipeline and save outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from point_e.util.point_cloud import PointCloud
from point_e.enhancements.advanced_enhancements import enhance_point_cloud_advanced

OUTPUT_DIR = Path("quick_demo_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Synthetic noisy geometry: cube edges + random jitter
def create_noisy_cube(num_points=2048, noise_level=0.02):
    points = []
    # edges of a cube [-0.8,0.8]^3
    coords = [-0.8, 0.8]
    for x in coords:
        for y in coords:
            for z in np.linspace(-0.8, 0.8, num_points // 8):
                points.append([x, y, z])
    for x in coords:
        for y in np.linspace(-0.8, 0.8, num_points // 8):
            for z in coords:
                points.append([x, y, z])
    for x in np.linspace(-0.8, 0.8, num_points // 8):
        for y in coords:
            for z in coords:
                points.append([x, y, z])

    points = np.array(points)[np.random.choice(len(points), num_points, replace=False)]
    points = points + np.random.randn(*points.shape) * noise_level
    colors = np.full((len(points), 3), 0.7)

    return PointCloud(coords=points, channels={"R": colors[:, 0], "G": colors[:, 1], "B": colors[:, 2]})

original_pc = create_noisy_cube()

enhanced_pc = enhance_point_cloud_advanced(
    original_pc,
    target_density=4096,
    densification_method="poisson",
    smooth_iterations=2,
    improve_structure=True,
)

# Validate
val_orig = original_pc.validate(min_points=512)
val_enh = enhanced_pc.validate(min_points=2048)
print("Original validity", val_orig)
print("Enhanced validity", val_enh)

# Save point clouds
original_pc.save(OUTPUT_DIR / "original_synthetic_cube.npz")
enhanced_pc.save(OUTPUT_DIR / "enhanced_synthetic_cube.npz")

# Plot comparison
def plot_pc(ax, pc, title):
    ax.scatter(pc.coords[:, 0], pc.coords[:, 1], pc.coords[:, 2], c=np.stack([pc.channels['R'], pc.channels['G'], pc.channels['B']], axis=1), s=1)
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.axis('off')

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_pc(ax1, original_pc, f"Original: {len(original_pc.coords)} pts")
plot_pc(ax2, enhanced_pc, f"Enhanced: {len(enhanced_pc.coords)} pts")
plt.tight_layout()
comparison_path = OUTPUT_DIR / "comparison_synthetic_cube.png"
fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Saved comparison image: {comparison_path}")
print("Done")
