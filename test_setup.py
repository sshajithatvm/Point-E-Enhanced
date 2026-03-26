#!/usr/bin/env python
"""
End-to-end test script for Point-E setup validation
Tests all major components without requiring large model downloads
"""

import sys
import traceback

def test_imports():
    """Test all critical imports"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        print("✓ torch")
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        print("✓ numpy")
        import numpy as np
        print(f"  NumPy version: {np.__version__}")
        
        print("✓ scipy")
        import scipy
        print(f"  SciPy version: {scipy.__version__}")
        
        print("✓ matplotlib")
        import matplotlib
        print(f"  Matplotlib version: {matplotlib.__version__}")
        
        print("✓ PIL")
        from PIL import Image
        print("  PIL imported successfully")
        
        print("✓ scikit-image")
        import skimage
        print(f"  scikit-image version: {skimage.__version__}")
        
        print("✓ open3d")
        import open3d as o3d
        print(f"  Open3D version: {o3d.__version__}")
        
        print("✓ clip")
        import clip
        print("  CLIP imported successfully")
        
        print("✓ tqdm")
        from tqdm.auto import tqdm
        print("  tqdm imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_point_e_modules():
    """Test Point-E specific modules"""
    print("\n" + "=" * 60)
    print("Testing Point-E modules...")
    print("=" * 60)
    
    try:
        import numpy as np
        print("✓ point_e.util.point_cloud")
        from point_e.util.point_cloud import PointCloud
        
        # Test creating a simple point cloud
        test_points = np.random.randn(100, 3).astype(np.float32)
        pc = PointCloud(test_points, {})
        print(f"  Created test PointCloud with {len(pc.coords)} points")
        
        print("✓ point_e.util.plotting")
        from point_e.util.plotting import plot_point_cloud
        
        print("✓ point_e.util.mesh")
        from point_e.util.mesh import TriMesh
        
        print("✓ point_e.util.pc_to_mesh")
        from point_e.util.pc_to_mesh import marching_cubes_mesh
        
        print("✓ point_e.diffusion.configs")
        from point_e.diffusion.configs import DIFFUSION_CONFIGS
        print(f"  Available diffusion configs: {list(DIFFUSION_CONFIGS.keys())}")
        
        print("✓ point_e.models.configs")
        from point_e.models.configs import MODEL_CONFIGS
        print(f"  Available model configs: {list(MODEL_CONFIGS.keys())}")
        
        print("✓ point_e.enhancements.applyenhancements")
        from point_e.enhancements.applyenhancements import enhance_point_cloud
        
        print("✓ point_e.evals.feature_extractor")
        from point_e.evals.feature_extractor import FeatureExtractor
        
        return True
    except Exception as e:
        print(f"✗ Module test failed: {e}")
        traceback.print_exc()
        return False

def test_torch_gpu():
    """Test PyTorch GPU availability"""
    print("\n" + "=" * 60)
    print("Testing PyTorch hardware...")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("  Note: GPU not available. Code will run on CPU (slower)")
        
        print(f"✓ PyTorch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        return True
    except Exception as e:
        print(f"✗ Hardware test failed: {e}")
        traceback.print_exc()
        return False

def test_jupyter():
    """Test Jupyter installation"""
    print("\n" + "=" * 60)
    print("Testing Jupyter...")
    print("=" * 60)
    
    try:
        print("✓ jupyter")
        import jupyter
        print("  Jupyter imported successfully")
        
        print("✓ jupyterlab")
        import jupyterlab
        print(f"  JupyterLab version: {jupyterlab.__version__}")
        
        print("✓ ipython")
        import IPython
        print(f"  IPython version: {IPython.__version__}")
        
        print("✓ ipykernel")
        import ipykernel
        print(f"  ipykernel version: {ipykernel.__version__}")
        
        return True
    except Exception as e:
        print(f"✗ Jupyter test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║ Point-E Environment Validation Test Suite".ljust(59) + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = {
        "Imports": test_imports(),
        "Point-E Modules": test_point_e_modules(),
        "PyTorch Hardware": test_torch_gpu(),
        "Jupyter": test_jupyter(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests PASSED! Environment is ready to use.")
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        print("   .\\venv\\Scripts\\activate.ps1")
        print("2. Launch JupyterLab:")
        print("   jupyter lab")
        print("3. Open and run the example notebooks:")
        print("   - point_e/examples/text2pointcloud.ipynb")
        print("   - point_e/examples/image2pointcloud.ipynb")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
