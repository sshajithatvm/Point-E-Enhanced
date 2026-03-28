#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import traceback

def test_imports():
    """Test all critical imports to ensure installation is working."""
    print("🧪 TESTING IMPORTS")
    print("=" * 50)
    
    tests = [
        ("Core Python", lambda: __import__('sys')),
        ("NumPy", lambda: __import__('numpy')),
        ("PyTorch", lambda: __import__('torch')),
        ("Matplotlib", lambda: __import__('matplotlib.pyplot')),
        ("SciPy", lambda: __import__('scipy')),
        ("scikit-image", lambda: __import__('skimage')),
        ("PIL/Pillow", lambda: __import__('PIL')),
        ("tqdm", lambda: __import__('tqdm')),
        ("requests", lambda: __import__('requests')),
        ("fire", lambda: __import__('fire')),
        ("humanize", lambda: __import__('humanize')),
        ("filelock", lambda: __import__('filelock')),
        ("Open3D", lambda: __import__('open3d')),
        ("CLIP", lambda: __import__('clip')),
    ]
    
    passed = 0
    failed = 0
    
    for name, test in tests:
        try:
            test()
            print(f"✅ {name}")
            passed += 1
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {name}: {e}")
            failed += 1
    
    print(f"\n📊 RESULTS: {passed} passed, {failed} failed")
    return failed == 0

def test_point_e_imports():
    """Test Point-E specific imports."""
    print("\n🔬 TESTING POINT-E IMPORTS")
    print("=" * 50)
    
    tests = [
        ("Point-E Core", lambda: __import__('point_e')),
        ("Point-E Models", lambda: __import__('point_e.models')),
        ("Point-E Diffusion", lambda: __import__('point_e.diffusion')),
        ("Point-E Utils", lambda: __import__('point_e.util')),
        ("Point-E Enhancements", lambda: __import__('point_e.enhancements')),
        ("Point-E Evaluation", lambda: __import__('point_e.evals')),
    ]
    
    passed = 0
    failed = 0
    
    for name, test in tests:
        try:
            test()
            print(f"✅ {name}")
            passed += 1
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {name}: {e}")
            failed += 1
    
    print(f"\n📊 RESULTS: {passed} passed, {failed} failed")
    return failed == 0

def test_enhancement_system():
    """Test the enhancement system specifically."""
    print("\n🔧 TESTING ENHANCEMENT SYSTEM")
    print("=" * 50)
    
    try:
        from point_e.enhancements.applyenhancements import enhance_point_cloud
        from point_e.enhancements.production_enhancement import enhance_point_cloud_production
        from point_e.util.point_cloud import PointCloud
        
        print("✅ Enhancement imports successful")
        
        # Test with a simple point cloud
        import numpy as np
        from point_e.util.point_cloud import PointCloud
        
        # Create test point cloud
        points = np.random.rand(100, 3)
        channels = {'R': np.random.rand(100), 'G': np.random.rand(100), 'B': np.random.rand(100)}
        test_pc = PointCloud(coords=points, channels=channels)
        
        print("✅ Test point cloud created")
        
        # Test legacy enhancement (using default parameters)
        legacy_enhanced = enhance_point_cloud(test_pc)
        print(f"✅ Legacy enhancement: {len(legacy_enhanced.coords)} points")
        
        # Test production enhancement
        production_enhanced = enhance_point_cloud_production(test_pc, prompt="test", save_ply=False, optimize_density=True)
        print(f"✅ Production enhancement: {len(production_enhanced.coords)} points")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhancement system error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic Point-E functionality."""
    print("\n⚙️  TESTING BASIC FUNCTIONALITY")
    print("=" * 50)
    
    try:
        from point_e.models.configs import MODEL_CONFIGS
        from point_e.diffusion.configs import DIFFUSION_CONFIGS
        from point_e.models.download import load_checkpoint
        from point_e.util.plotting import plot_point_cloud
        from point_e.util.point_cloud import PointCloud
        
        print("✅ Core Point-E imports successful")
        
        # Test configs
        print(f"✅ Available models: {list(MODEL_CONFIGS.keys())}")
        print(f"✅ Available diffusions: {list(DIFFUSION_CONFIGS.keys())}")
        
        # Test plotting
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create a simple test plot
        import numpy as np
        points = np.random.rand(50, 3)
        channels = {'R': np.random.rand(50), 'G': np.random.rand(50), 'B': np.random.rand(50)}
        test_pc = PointCloud(coords=points, channels=channels)
        
        fig = plot_point_cloud(test_pc, grid_size=1)
        print("✅ Basic plotting functionality works")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 COMPREHENSIVE POINT-E INSTALLATION TEST")
    print("=" * 60)
    
    # Test basic imports
    imports_ok = test_imports()
    
    # Test Point-E imports
    pointe_imports_ok = test_point_e_imports()
    
    # Test enhancement system
    enhancement_ok = test_enhancement_system()
    
    # Test basic functionality
    basic_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 60)
    
    all_tests = [
        ("Basic Dependencies", imports_ok),
        ("Point-E Imports", pointe_imports_ok),
        ("Enhancement System", enhancement_ok),
        ("Basic Functionality", basic_ok),
    ]
    
    passed = 0
    failed = 0
    
    for name, result in all_tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:<25} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{passed + failed} tests passed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Installation is successful.")
        print("\n📝 NEXT STEPS:")
        print("1. Run 'jupyter lab' to start JupyterLab")
        print("2. Open point_e/examples/text2pointcloud.ipynb")
        print("3. Run the notebook cells to test end-to-end functionality")
        return True
    else:
        print("❌ SOME TESTS FAILED! Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
