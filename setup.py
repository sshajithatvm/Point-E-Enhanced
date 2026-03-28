from setuptools import setup

setup(
    name="point-e",
    version="1.0.0-optimized",
    description="Point-E: System for Generating 3D Point Clouds (Production-Optimized)",
    packages=[
        "point_e",
        "point_e.diffusion",
        "point_e.evals",
        "point_e.models",
        "point_e.util",
        "point_e.enhancements",
        "point_e.optimization",
    ],
    install_requires=[
        "filelock>=3.0.0",
        "Pillow>=8.0.0",
        "torch>=1.9.0",
        "fire>=0.4.0",
        "humanize>=3.0.0",
        "requests>=2.25.0",
        "tqdm>=4.50.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.18.0",
        "scipy>=1.5.0",
        "numpy>=1.19.0",
        "open3d>=0.13.0",
        "psutil>=5.8.0",
        "clip @ git+https://github.com/openai/CLIP.git",
    ],
    python_requires=">=3.8",
    author="OpenAI (Production Optimizations by Assistant)",
)
