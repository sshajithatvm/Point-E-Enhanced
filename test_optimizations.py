"""
Comprehensive test suite for optimized Point-E.
Tests all components, error handling, and end-to-end functionality.
"""

import unittest
import tempfile
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPerformanceOptimizer(unittest.TestCase):
    """Test PerformanceOptimizer functionality."""
    
    def setUp(self):
        from point_e.optimization.performance_optimizer import (
            PerformanceOptimizer,
            OptimizationConfig,
            CLIPEmbeddingCache,
        )
        self.PerformanceOptimizer = PerformanceOptimizer
        self.OptimizationConfig = OptimizationConfig
        self.CLIPEmbeddingCache = CLIPEmbeddingCache
    
    def test_optimizer_initialization(self):
        """Test optimizer can be initialized."""
        config = self.OptimizationConfig(batch_size=4)
        optimizer = self.PerformanceOptimizer(config)
        self.assertEqual(optimizer.config.batch_size, 4)
        logger.info("✓ Optimizer initialization test passed")
    
    def test_batch_creation(self):
        """Test batch creation."""
        optimizer = self.PerformanceOptimizer()
        prompts = ["a", "b", "c", "d", "e", "f"]
        batches = optimizer.create_batches(prompts)
        self.assertEqual(len(batches), 2)  # 6 prompts, batch_size=4 → 2 batches
        logger.info("✓ Batch creation test passed")
    
    def test_clip_cache(self):
        """Test CLIP embedding cache."""
        cache = self.CLIPEmbeddingCache(max_size=10)
        
        # Test cache miss
        embedding = torch.randn(512)
        cache.put("test_prompt", embedding)
        retrieved = cache.get("test_prompt")
        
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.allclose(retrieved, embedding))
        
        stats = cache.get_stats()
        self.assertEqual(stats["cache_size"], 1)
        self.assertEqual(stats["cache_hits"], 1)
        
        logger.info("✓ CLIP cache test passed")
    
    def test_karras_steps_optimization(self):
        """Test karras steps optimization."""
        optimizer = self.PerformanceOptimizer()
        original_steps = (64, 64)
        optimized = optimizer.optimize_karras_steps(original_steps, reduction_factor=0.75)
        
        self.assertEqual(optimized, (48, 48))  # 64 * 0.75 = 48
        logger.info("✓ Karras steps optimization test passed")


class TestLogger(unittest.TestCase):
    """Test logging functionality."""
    
    def setUp(self):
        from point_e.optimization.logger import (
            PointELogger,
            MetricsLogger,
            PerformanceContext,
        )
        self.PointELogger = PointELogger
        self.MetricsLogger = MetricsLogger
        self.PerformanceContext = PerformanceContext
    
    def test_logger_initialization(self):
        """Test logger can be initialized."""
        logger_instance = self.PointELogger.get_logger("test")
        self.assertIsNotNone(logger_instance)
        logger.info("✓ Logger initialization test passed")
    
    def test_metrics_logger(self):
        """Test metrics logging."""
        metrics = self.MetricsLogger("test_metrics")
        
        # Should not raise
        metrics.log_timing("test_operation", 1.5, items_processed=10)
        metrics.log_memory_usage("test_memory", 256.5)
        
        logger.info("✓ Metrics logging test passed")
    
    def test_performance_context(self):
        """Test performance context manager."""
        import time
        
        with self.PerformanceContext("test_context") as ctx:
            time.sleep(0.1)
        
        self.assertGreaterEqual(ctx.start_time, 0)
        logger.info("✓ Performance context test passed")


class TestBilateralSmoothing(unittest.TestCase):
    """Test optimized bilateral smoothing."""
    
    def setUp(self):
        from point_e.enhancements.bilateralsmoothing import bilateral_smoothing
        self.bilateral_smoothing = bilateral_smoothing
    
    def test_bilateral_smoothing_without_crash(self):
        """Test bilateral smoothing doesn't crash."""
        # Create simple test point cloud
        np.random.seed(42)
        points = np.random.randn(100, 3)
        
        # Should not raise
        smoothed = self.bilateral_smoothing(
            points,
            spatial_sigma=0.1,
            intensity_sigma=0.1,
            radius=0.2,
            use_kdtree=True,
            num_iterations=1,
        )
        
        self.assertEqual(smoothed.shape, points.shape)
        logger.info("✓ Bilateral smoothing test passed")


class TestAdvancedEnhancements(unittest.TestCase):
    """Test advanced point cloud enhancements."""
    
    def setUp(self):
        from point_e.enhancements.advanced_enhancements import (
            densify_point_cloud,
            improve_structural_accuracy,
        )
        self.densify_point_cloud = densify_point_cloud
        self.improve_structural_accuracy = improve_structural_accuracy
    
    def test_densification(self):
        """Test point cloud densification."""
        np.random.seed(42)
        points = np.random.randn(100, 3)
        
        # Test upsampling via interpolation
        densified = self.densify_point_cloud(
            points,
            target_point_count=200,
            method="interpolation",
        )
        
        self.assertGreaterEqual(len(densified), 200)
        logger.info(f"✓ Densification test passed ({len(points)} → {len(densified)} points)")
    
    def test_structural_accuracy(self):
        """Test structural accuracy improvement."""
        np.random.seed(42)
        points = np.random.randn(100, 3)
        
        # Should not raise
        improved = self.improve_structural_accuracy(
            points,
            smoothing_iterations=1,
            outlier_removal=True,
        )
        
        self.assertLessEqual(len(improved), len(points))  # May remove outliers
        logger.info(f"✓ Structural accuracy test passed")


class TestProductionGenerator(unittest.TestCase):
    """Test production generator."""
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_generator_initialization(self):
        """Test generator can be initialized."""
        from point_e_production import ProductionPointEGenerator
        
        gen = ProductionPointEGenerator(
            device=torch.device("cpu"),
            output_dir=self.output_dir,
        )
        self.assertIsNotNone(gen)
        logger.info("✓ Generator initialization test passed")
    
    def test_performance_report(self):
        """Test performance report generation."""
        from point_e_production import ProductionPointEGenerator
        
        gen = ProductionPointEGenerator(
            device=torch.device("cpu"),
            output_dir=self.output_dir,
        )
        report = gen.get_performance_report()
        self.assertIsInstance(report, str)
        self.assertIn("PERFORMANCE", report)
        logger.info("✓ Performance report test passed")


class IntegrationTests(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_point_cloud_object_creation(self):
        """Test creating PointCloud objects."""
        from point_e.util.point_cloud import PointCloud
        
        # Create simple point cloud
        coords = np.random.randn(100, 3)
        channels = {"R": np.ones(100), "G": np.ones(100), "B": np.ones(100)}
        
        pc = PointCloud(coords=coords, channels=channels)
        self.assertEqual(len(pc.coords), 100)
        
        # Test save/load
        path = self.output_dir / "test.npz"
        pc.save(path)
        loaded_pc = PointCloud.load(path)
        
        self.assertEqual(len(loaded_pc.coords), len(pc.coords))
        logger.info("✓ PointCloud creation/save/load test passed")
    
    def test_optimization_pipeline_no_crash(self):
        """Test optimization pipeline doesn't crash."""
        from point_e.optimization.performance_optimizer import create_default_optimizer
        from point_e.optimization.logger import MetricsLogger
        
        # Should not raise
        optimizer = create_default_optimizer()
        metrics = MetricsLogger()
        
        prompts = ["red ball", "blue cube"]
        batches = optimizer.create_batches(prompts)
        self.assertEqual(len(batches), 1)  # 2 prompts, default batch_size=4
        
        logger.info("✓ Optimization pipeline test passed")


def run_all_tests():
    """Run complete test suite."""
    print("=" * 70)
    print("POINT-E OPTIMIZATION TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestBilateralSmoothing))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedEnhancements))
    suite.addTests(loader.loadTestsFromTestCase(TestProductionGenerator))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
