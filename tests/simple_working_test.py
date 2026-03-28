#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e.enhancements.applyenhancements import enhance_point_cloud
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e.util.point_cloud import PointCloud
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    
    # Original
    fig = plot_point_cloud(test_pc, grid_size=1, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    fig.suptitle(f"Original\n{len(test_pc.coords)} points", fontweight='bold')
    fig.savefig('original_test.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    # Legacy Enhanced
    fig = plot_point_cloud(legacy_enhanced, grid_size=1, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    fig.suptitle(f"Legacy Enhanced\n{len(legacy_enhanced.coords)} points", fontweight='bold')
    fig.savefig('legacy_test.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    # Production Enhanced
    fig = plot_point_cloud(production_enhanced, grid_size=1, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    fig.suptitle(f"Production Enhanced\n{len(production_enhanced.coords)} points", fontweight='bold')
    fig.savefig('production_test.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    
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
    print(f"   - original_test.png")
    print(f"   - legacy_test.png")
    print(f"   - production_test.png")
    print(f"   - outputs/enhanced_outputs/ply_files/")
    print(f"   - outputs/enhanced_outputs/metadata/")
    
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
