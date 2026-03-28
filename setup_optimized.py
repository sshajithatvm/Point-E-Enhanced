from setuptools import setup, find_packages

setup(
    name="point-e-optimized",
    version="2.0.0",
    description="High-performance, production-ready Point-E point cloud generation",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "scikit-image>=0.19.0",
        "scikit-learn>=1.0.0",
        "open3d>=0.15.0",
        "psutil>=5.8.0",
        "clip @ git+https://github.com/openai/CLIP.git",
        "tqdm>=4.60.0",
        "requests>=2.25.0",
        "filelock>=3.20.0",
        "Pillow>=9.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
