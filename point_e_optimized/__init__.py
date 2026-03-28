"""
Point-E Optimized: High-performance, production-ready point cloud generation system.

This module provides optimized point cloud generation with:
- Multi-processing support for maximum CPU utilization
- Batched processing for improved throughput
- Enhanced Open3D post-processing
- Modular architecture with robust error handling
- Structured logging and performance monitoring
"""

from .core import PointEGenerator
from .processors import PointCloudProcessor
from .visualizers import PointCloudVisualizer
from .utils import setup_logging, PerformanceMonitor

__version__ = "2.0.0"
__all__ = ["PointEGenerator", "PointCloudProcessor", "PointCloudVisualizer", "setup_logging", "PerformanceMonitor"]
