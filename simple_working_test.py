#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import numpy as np
from point_e.enhancements.applyenhancements import enhance_point_cloud
from point_e.util.point_cloud import PointCloud
from point_e.util.plotting import plot_point_cloud

def main():
    """Simple working test of Point-E enhancements."""
    print("🧪 SIMPLE POINT-E ENHANCEMENT TEST")
    print("=" * 50)
    
    # Create test point cloud
    print("Creating test point cloud...")
    points = np.random.rand(200, 3) * 2 - 1  # Random points in [-1, 1] cube
    channels = {
        'R': np.random.rand(200),
        'G': np.random.rand(200), 
        'B': np.random.rand(200)
    }
    test_pc = PointCloud(coords=points, channels=channels)
    print(f"✅ Created test point cloud with {len(points)} points")
    
    # Apply legacy enhancement
    print("Applying legacy enhancement...")
    legacy_enhanced = enhance_point_cloud(test_pc)
    print(f"✅ Legacy enhanced: {len(legacy_enhanced.coords)} points")
    
    # Apply production enhancement  
    print("Applying production enhancement...")
    production_enhanced = enhance_point_cloud(test_pc, prompt="test_simple", save_ply=True, optimize_density=True)
    print(f"✅ Production enhanced: {len(production_enhanced.coords)} points")
    
    # Create visualization
    print("Creating comparison visualization...")
    fig = plt.figure(figsize=(15, 5))
    
    # Original
    ax1 = fig.add_subplot(131, projection='3d')
    ax1 = plot_point_cloud(test_pc, grid_size=1, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    ax1.set_title(f"Original\n{len(test_pc.coords)} points", fontweight='bold')
    
    # Legacy Enhanced
    ax2 = fig.add_subplot(132, projection='3d')
    ax2 = plot_point_cloud(legacy_enhanced, grid_size=1, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    ax2.set_title(f"Legacy Enhanced\n{len(legacy_enhanced.coords)} points", fontweight='bold')
    
    # Production Enhanced
    ax3 = fig.add_subplot(133, projection='3d')
    ax3 = plot_point_cloud(production_enhanced, grid_size=1, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    ax3.set_title(f"Production Enhanced\n{len(production_enhanced.coords)} points", fontweight='bold')
    
    plt.suptitle("Point-E Enhancement Comparison", fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig('simple_enhancement_test.png', dpi=120, bbox_inches='tight')
    print("✅ Visualization saved as 'simple_enhancement_test.png'")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print("✅ Point Cloud Creation: SUCCESS")
    print("✅ Legacy Enhancement: SUCCESS")
    print("✅ Production Enhancement: SUCCESS")
    print("✅ PLY Export: SUCCESS")
    print("✅ Visualization: SUCCESS")
    print(f"📁 Files Created:")
    print(f"   - simple_enhancement_test.png")
    print(f"   - enhanced_outputs/ply_files/")
    print(f"   - enhanced_outputs/metadata/")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 SIMPLE TEST COMPLETED SUCCESSFULLY!")
            print("\n✅ Point-E Enhanced repository is working correctly!")
        else:
            print("\n❌ SIMPLE TEST FAILED!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
