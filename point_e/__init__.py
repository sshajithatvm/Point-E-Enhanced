"""
Point-E: A System for Generating 3D Point Clouds from Complex Prompts.
Production-Optimized Version with Performance Enhancements.

This package provides:
- Text-to-3D and Image-to-3D point cloud generation
- High-performance optimization framework
- Advanced point cloud post-processing
- Structured logging and benchmarking
"""

__version__ = "1.0.0-optimized"
__author__ = "OpenAI (Production Optimizations)"

# Core imports
from . import diffusion
from . import models
from . import util
from . import evals

# Enhancement modules
from . import enhancements
from . import optimization

# Make key classes available at package level
from .util.point_cloud import PointCloud
from .diffusion.sampler import PointCloudSampler
from .optimization.performance_optimizer import PerformanceOptimizer, OptimizationConfig
from .optimization.logger import PointELogger, MetricsLogger
from .optimization.benchmarking import PerformanceBenchmark

__all__ = [
    "PointCloud",
    "PointCloudSampler",
    "PerformanceOptimizer",
    "OptimizationConfig",
    "PointELogger",
    "MetricsLogger",
    "PerformanceBenchmark",
    "diffusion",
    "models",
    "util",
    "evals",
    "enhancements",
    "optimization",
]
