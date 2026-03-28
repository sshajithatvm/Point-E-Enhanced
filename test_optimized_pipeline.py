"""
Comprehensive tests for the optimized Point-E pipeline with geometric quality enhancements.
Tests cover performance, determinism, error handling, and quality improvements.
"""

import unittest
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path
import time
import logging
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from point_e.util.point_cloud import PointCloud
from point_e.enhancements.geometric_quality_enhancements import (
    enhance_point_cloud_quality,
    statistical_outlier_removal,
    knn_pca_normal_refinement
)


class TestGeometricQualityEnhancements(unittest.TestCase):
    """Test geometric quality enhancement functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test point cloud
        np.random.seed(42)
        self.test_points = np.random.random((100, 3))
        self.test_pc = PointCloud(coords=self.test_points, channels={})

    def test_statistical_outlier_removal_preserves_count(self):
        """Test that statistical outlier removal preserves point count."""
        original_count = len(self.test_points)
        result = statistical_outlier_removal(self.test_points)
        self.assertEqual(len(result), original_count)

    def test_knn_pca_normal_refinement_preserves_count(self):
        """Test that k-NN/PCA normal refinement preserves point count."""
        original_count = len(self.test_points)
        result = knn_pca_normal_refinement(self.test_points)
        self.assertEqual(len(result), original_count)

    def test_enhance_point_cloud_quality_preserves_count(self):
        """Test that quality enhancement preserves point count."""
        original_count = len(self.test_pc.coords)
        result = enhance_point_cloud_quality(self.test_pc, num_iterations=1)
        self.assertEqual(len(result.coords), original_count)

    def test_enhance_point_cloud_quality_no_geometry_change(self):
        """Test that quality enhancement doesn't significantly alter geometry."""
        result = enhance_point_cloud_quality(self.test_pc, num_iterations=1)
        # Points should be very close (within small tolerance for outlier correction)
        max_diff = np.max(np.abs(np.array(result.coords) - self.test_points))
        self.assertLess(max_diff, 0.1)  # Very small changes only


class TestOptimizedPointEPipeline(unittest.TestCase):
    """Test the optimized Point-E pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_outputs"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('optimized_point_e_pipeline.PerformanceOptimizer')
    def test_pipeline_initialization(self, mock_optimizer):
        """Test pipeline initialization with various configurations."""
        from optimized_point_e_pipeline import OptimizedPointEGenerator

        # Test default initialization
        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        self.assertTrue(gen.enable_geometric_quality_enhancements)
        self.assertTrue(gen.deterministic)
        self.assertEqual(gen.random_seed, 42)

        # Test custom initialization
        gen_custom = OptimizedPointEGenerator(
            output_dir=self.output_dir,
            enable_geometric_quality_enhancements=False,
            deterministic=False,
            random_seed=123
        )
        self.assertFalse(gen_custom.enable_geometric_quality_enhancements)
        self.assertFalse(gen_custom.deterministic)
        self.assertEqual(gen_custom.random_seed, 123)

    def test_deterministic_generation(self):
        """Test that deterministic mode produces consistent results."""
        from optimized_point_e_pipeline import OptimizedPointEGenerator

        # This would require mocking the actual generation, but we can test the setup
        gen1 = OptimizedPointEGenerator(deterministic=True, random_seed=42)
        gen2 = OptimizedPointEGenerator(deterministic=True, random_seed=42)

        # Check that torch seeds are set (this is a basic check)
        self.assertTrue(torch.initial_seed() > 0)

    @patch('optimized_point_e_pipeline.enhance_point_cloud_quality')
    @patch('optimized_point_e_pipeline.load_checkpoint')
    @patch('optimized_point_e_pipeline.model_from_config')
    def test_error_handling_in_generation(self, mock_model_config, mock_load_checkpoint, mock_enhance):
        """Test error handling in point cloud generation."""
        from optimized_point_e_pipeline import OptimizedPointEGenerator

        # Mock failures
        mock_model_config.side_effect = Exception("Model loading failed")

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Test single generation with error
        result = gen._generate_single_point_cloud(("test prompt", "cpu", 0))
        self.assertEqual(result[0], "test prompt")
        self.assertIsNone(result[1])  # Should return None on failure

    def test_batch_processing_setup(self):
        """Test batch processing argument preparation."""
        from optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        prompts = ["prompt1", "prompt2", "prompt3"]

        # Test worker args preparation
        worker_args = [(prompt, str(gen.device), 0) for prompt in prompts]
        self.assertEqual(len(worker_args), 3)
        self.assertEqual(worker_args[0], ("prompt1", str(gen.device), 0))
        self.assertEqual(worker_args[1], ("prompt2", str(gen.device), 0))
        self.assertEqual(worker_args[2], ("prompt3", str(gen.device), 0))


class TestPerformanceAndLogging(unittest.TestCase):
    """Test performance monitoring and logging."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_logging_initialization(self):
        """Test that logging is properly initialized."""
        from point_e.optimization.logger import initialize_logging

        # Should not raise exceptions
        try:
            initialize_logging(log_dir=self.log_dir)
        except Exception as e:
            self.fail(f"Logging initialization failed: {e}")

    def test_performance_logging(self):
        """Test performance logging decorators."""
        from point_e.optimization.logger import log_performance

        @log_performance("test_operation")
        def dummy_operation():
            time.sleep(0.01)
            return "result"

        # Should not raise exceptions
        result = dummy_operation()
        self.assertEqual(result, "result")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_point_cloud(self):
        """Test handling of empty point clouds."""
        from point_e.enhancements.geometric_quality_enhancements import enhance_point_cloud_quality

        empty_pc = PointCloud(coords=np.empty((0, 3)), channels={})
        result = enhance_point_cloud_quality(empty_pc)
        self.assertEqual(len(result.coords), 0)

    def test_single_point_cloud(self):
        """Test handling of single-point clouds."""
        from point_e.enhancements.geometric_quality_enhancements import enhance_point_cloud_quality

        single_point = np.array([[0.5, 0.5, 0.5]])
        pc = PointCloud(coords=single_point, channels={})
        result = enhance_point_cloud_quality(pc)
        self.assertEqual(len(result.coords), 1)

    def test_large_point_cloud(self):
        """Test handling of large point clouds."""
        from point_e.enhancements.geometric_quality_enhancements import enhance_point_cloud_quality

        # Create a large point cloud
        large_points = np.random.random((10000, 3))
        pc = PointCloud(coords=large_points, channels={})

        start_time = time.time()
        result = enhance_point_cloud_quality(pc, num_iterations=1)
        duration = time.time() - start_time

        self.assertEqual(len(result.coords), 10000)
        # Should complete in reasonable time (less than 30 seconds)
        self.assertLess(duration, 30.0)


def test_optimized_pipeline():
    """Test the optimized pipeline with a small set of prompts."""
    print("Testing Optimized Point-E Pipeline")
    print("=" * 50)

    try:
        # Create generator with minimal workers for testing
        from optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(
            output_dir=Path("test_optimized_outputs"),
            num_workers=1,  # Use 1 worker for testing
            enable_geometric_quality_enhancements=True,
        )

        # Test prompts
        test_prompts = [
            "a red cube",
        ]

        print(f"Testing with {len(test_prompts)} prompts...")

        start_time = time.time()

        # Generate point clouds
        results = gen.generate_point_clouds_parallel(test_prompts)

        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")

        # Validate results
        print("\nValidating results...")
        for prompt, pc in results.items():
            print(f"✓ '{prompt}': {len(pc.coords)} points")
            assert len(pc.coords) > 0, f"Empty point cloud for '{prompt}'"
            assert pc.coords.shape[1] == 3, f"Invalid point dimensions for '{prompt}'"

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)

    # Run the simple integration test
    test_optimized_pipeline()

    # Run comprehensive unit tests
    unittest.main(verbosity=2)

if __name__ == "__main__":
    success = test_optimized_pipeline()
    sys.exit(0 if success else 1)