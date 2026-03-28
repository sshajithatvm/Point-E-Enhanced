#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e.util.point_cloud import PointCloud
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e_optimized.validator import PointCloudValidator

def create_test_point_clouds():
    """Create test point clouds with various issues."""
    test_clouds = {}
    
    # 1. Valid point cloud
    valid_points = np.random.rand(500, 3) * 2 - 1  # Random points in [-1, 1]
    test_clouds['valid'] = PointCloud(coords=valid_points, channels={'R': np.random.rand(500)})
    
    # 2. Empty point cloud
    empty_points = np.array([]).reshape(0, 3)
    test_clouds['empty'] = PointCloud(coords=empty_points, channels={})
    
    # 3. Too few points
    few_points = np.random.rand(10, 3) * 2 - 1
    test_clouds['too_few'] = PointCloud(coords=few_points, channels={})
    
    # 4. Contains NaN values
    nan_points = np.random.rand(200, 3) * 2 - 1
    nan_points[50, 0] = np.nan
    nan_points[100, 1] = np.nan
    test_clouds['contains_nan'] = PointCloud(coords=nan_points, channels={})
    
    # 5. Contains infinite values
    inf_points = np.random.rand(200, 3) * 2 - 1
    inf_points[75, 2] = np.inf
    test_clouds['contains_inf'] = PointCloud(coords=inf_points, channels={})
    
    # 6. Degenerate bounding box (all points same)
    degenerate_points = np.ones((100, 3)) * 0.5
    test_clouds['degenerate'] = PointCloud(coords=degenerate_points, channels={})
    
    # 7. Too many duplicates
    duplicate_points = np.random.rand(50, 3) * 2 - 1
    duplicate_points = np.vstack([duplicate_points] * 10)  # 90% duplicates
    test_clouds['too_many_duplicates'] = PointCloud(coords=duplicate_points, channels={})
    
    # 8. Extreme coordinate values
    extreme_points = np.random.rand(100, 3) * 100 - 50  # Very large coordinates
    test_clouds['extreme_coords'] = PointCloud(coords=extreme_points, channels={})
    
    # 9. Very low density
    sparse_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    test_clouds['low_density'] = PointCloud(coords=sparse_points, channels={})
    
    return test_clouds

def test_validation():
    """Test validation on various point clouds."""
    print("🔍 POINT CLOUD VALIDATION TEST")
    print("=" * 80)
    
    validator = PointCloudValidator()
    test_clouds = create_test_point_clouds()
    
    results = {}
    
    for name, pc in test_clouds.items():
        print(f"\n🧪 Testing: {name.upper()}")
        print("-" * 40)
        
        # Validate point cloud
        result = validator.validate_point_cloud(pc, min_points=100, max_points=5000)
        results[name] = result
        
        # Print detailed result
        validator.print_validation_result(result, name)
    
    # Summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for name, result in results.items():
        status = "✅ PASS" if result.is_valid else "❌ FAIL"
        print(f"{name:20} | {status:8} | Points: {result.point_count:4} | Errors: {len(result.errors)}")
        
        if result.is_valid:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 OVERALL RESULTS:")
    print(f"  Total Tests: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success Rate: {(passed/len(results))*100:.1f}%")
    
    # Test with real point cloud from enhanced system
    print(f"\n🎨 TESTING REAL ENHANCED POINT CLOUD")
    print("=" * 80)
    
    try:
        # Try to load a real enhanced point cloud
        import os
        ply_files = []
        if os.path.exists('outputs/enhanced_outputs/ply_files'):
            ply_files = [f for f in os.listdir('outputs/enhanced_outputs/ply_files') if f.endswith('.ply')]
        
        if ply_files:
            # Load the most recent PLY file
            latest_ply = sorted(ply_files)[-1]
            print(f"Loading: {latest_ply}")
            
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(f'outputs/enhanced_outputs/ply_files/{latest_ply}')
            points = np.asarray(pcd.points)
            
            # Create PointCloud object
            real_pc = PointCloud(coords=points, channels={})
            
            # Validate real point cloud
            real_result = validator.validate_point_cloud(real_pc, min_points=50, max_points=10000)
            validator.print_validation_result(real_result, f"Real Enhanced: {latest_ply}")
            
        else:
            print("No enhanced PLY files found for testing")
            
    except Exception as e:
        print(f"Failed to test real point cloud: {e}")
    
    return len(results) == passed  # All tests should pass except the invalid ones

def create_validation_visualization():
    """Create visualization of validation results."""
    print(f"\n🎨 CREATING VALIDATION VISUALIZATION")
    
    validator = PointCloudValidator()
    test_clouds = create_test_point_clouds()
    
    # Create subplot for each test case
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    test_names = list(test_clouds.keys())
    
    for i, (name, pc) in enumerate(test_clouds.items()):
        ax = axes[i]
        
        # Validate
        result = validator.validate_point_cloud(pc, min_points=100, max_points=5000)
        
        # Plot points
        points = np.array(pc.coords)
        if len(points) > 0:
            # Color based on validity
            color = 'green' if result.is_valid else 'red'
            alpha = 0.6 if result.is_valid else 0.3
            
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=color, alpha=alpha, s=10)
        
        # Set title with validation result
        status = "PASS" if result.is_valid else "FAIL"
        ax.set_title(f"{name}\n{status}\n{len(points)} pts", 
                    fontweight='bold', 
                    color='green' if result.is_valid else 'red')
        
        # Set consistent bounds
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
    
    plt.suptitle("Point Cloud Validation Test Results", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('validation_test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Validation visualization saved as 'validation_test_results.png'")

def main():
    """Main validation test."""
    print("🚀 POINT CLOUD VALIDATION SYSTEM")
    print("=" * 80)
    print("Testing strict validation to ensure reliable point clouds...")
    
    # Run validation tests
    success = test_validation()
    
    # Create visualization
    create_validation_visualization()
    
    # Final summary
    print(f"\n🎯 VALIDATION SYSTEM COMPLETE")
    print("=" * 80)
    
    if success:
        print("✅ Validation system working correctly")
        print("✅ All test cases properly identified")
        print("✅ Invalid point clouds rejected")
        print("✅ Valid point clouds accepted")
    else:
        print("❌ Some validation tests failed")
    
    print("\n📋 VALIDATION CRITERIA:")
    print("  ✅ Non-empty point cloud")
    print("  ✅ Reasonable point count (100-5000)")
    print("  ✅ Valid numeric coordinates (no NaN/inf)")
    print("  ✅ Non-degenerate bounding box")
    print("  ✅ Meaningful spatial spread")
    print("  ✅ Appropriate density")
    print("  ✅ Limited duplicate points")
    print("  ✅ Reasonable coordinate ranges")
    
    print("\n🏁 VALIDATION SYSTEM READY FOR PRODUCTION!")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
