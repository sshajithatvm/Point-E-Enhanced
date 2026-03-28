#!/usr/bin/env python
"""
End-to-End Point-E Production Demo
Runs in fresh environments with automatic setup and verification.

This script:
1. Installs/validates all dependencies
2. Generates point clouds with original and enhanced methods
3. Creates side-by-side comparison images
4. Provides performance benchmarks
5. Saves all outputs to production_comparison_results/
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def run_command(cmd, description, timeout=300):
    """Run a command with error handling."""
    print(f"\n🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"❌ Failed: {result.stderr}")
            return False
        print(f"✅ Success")
        return True
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")

    # Core dependencies
    deps = [
        "pip install --upgrade pip",
        "pip install torch>=1.9.0",
        "pip install open3d>=0.13.0",
        "pip install matplotlib>=3.3.0",
        "pip install scikit-image>=0.18.0",
        "pip install psutil>=5.8.0",
        "pip install pandas>=1.1.0",
        "pip install filelock Pillow fire humanize requests tqdm numpy scipy",
        "pip install git+https://github.com/openai/CLIP.git",
    ]

    for dep in deps:
        if not run_command(dep, f"Installing {dep.split()[-1]}"):
            return False

    # Install Point-E in development mode
    if not run_command("pip install -e .", "Installing Point-E (development mode)"):
        return False

    return True

def validate_installation():
    """Validate that all components are properly installed."""
    print("\n🔍 Validating installation...")

    validations = [
        ("import torch", "PyTorch"),
        ("import open3d as o3d", "Open3D"),
        ("import point_e", "Point-E"),
        ("from point_e.enhancements.advanced_enhancements import enhance_point_cloud_advanced", "Advanced Enhancements"),
        ("from point_e.optimization.performance_optimizer import create_default_optimizer", "Performance Optimizer"),
    ]

    for import_stmt, name in validations:
        try:
            exec(import_stmt)
            print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name} - Failed: {e}")
            return False

    return True

def run_production_demo():
    """Run the production comparison demo."""
    print("\n🚀 Running production comparison demo...")

    # Import and run the demo
    try:
        from production_comparison_demo import main
        main()
        return True
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main end-to-end execution."""
    print("=" * 80)
    print("Point-E Production End-to-End Demo")
    print("=" * 80)
    print("This script will:")
    print("1. Validate Python version")
    print("2. Install all dependencies")
    print("3. Validate installation")
    print("4. Run production comparison demo")
    print("5. Generate side-by-side comparison images")
    print("=" * 80)

    start_time = time.time()

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed")
        sys.exit(1)

    # Validate installation
    if not validate_installation():
        print("\n❌ Installation validation failed")
        sys.exit(1)

    # Run demo
    if not run_production_demo():
        print("\n❌ Production demo failed")
        sys.exit(1)

    # Success
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 SUCCESS! End-to-end demo completed")
    print("=" * 80)
    print(".1f")
    print("\n📁 Outputs saved to: production_comparison_results/")
    print("🖼️  Comparison images: production_comparison_results/comparison_images/")
    print("📊 Performance report: production_comparison_results/performance_report.json")
    print("\n🚀 Ready for production deployment!")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Projects\Point-E-Enhanced-Cline-Model\end_to_end_demo.py