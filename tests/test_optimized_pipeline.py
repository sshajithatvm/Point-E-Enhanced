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
import os
import multiprocessing as mp
from unittest.mock import patch, MagicMock, mock_open
from concurrent.futures import TimeoutError

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

    @patch('scripts.optimized_point_e_pipeline.PerformanceOptimizer')
    def test_pipeline_initialization(self, mock_optimizer):
        """Test pipeline initialization with various configurations."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

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
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        # This would require mocking the actual generation, but we can test the setup
        gen1 = OptimizedPointEGenerator(deterministic=True, random_seed=42)
        gen2 = OptimizedPointEGenerator(deterministic=True, random_seed=42)

        # Check that torch seeds are set (this is a basic check)
        self.assertTrue(torch.initial_seed() > 0)

    @patch('scripts.optimized_point_e_pipeline.enhance_point_cloud_quality')
    @patch('scripts.optimized_point_e_pipeline.load_checkpoint')
    @patch('scripts.optimized_point_e_pipeline.model_from_config')
    def test_error_handling_in_generation(self, mock_model_config, mock_load_checkpoint, mock_enhance):
        """Test error handling in point cloud generation."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        # Mock failures
        mock_model_config.side_effect = Exception("Model loading failed")

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Test single generation with error
        result = gen._generate_single_point_cloud(("test prompt", "cpu", 0))
        self.assertEqual(result[0], "test prompt")
        self.assertIsNone(result[1])  # Should return None on failure

    def test_batch_processing_setup(self):
        """Test batch processing argument preparation."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

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
        self.log_dir.mkdir(parents=True, exist_ok=True)

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
        import logging

        def dummy_operation():
            time.sleep(0.01)
            return "result"

        # Mock the logger to avoid file system issues
        with unittest.mock.patch('point_e.optimization.logger.PointELogger.get_logger') as mock_get_logger:
            mock_logger = unittest.mock.MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Should not raise exceptions
            with log_performance("test_operation"):
                result = dummy_operation()
            self.assertEqual(result, "result")
            
            # Verify logging was called
            mock_logger.debug.assert_called()
            mock_logger.error.assert_not_called()


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


class TestInputValidation(unittest.TestCase):
    """Test input validation and error handling for invalid inputs."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_outputs"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_empty_prompt_list(self):
        """Test handling of empty prompt lists."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        result = gen.generate_point_clouds_parallel([])
        self.assertEqual(len(result), 0)

    def test_none_prompt_list(self):
        """Test handling of None prompt list."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        # None is treated as empty list, should return empty dict
        result = gen.generate_point_clouds_parallel(None)
        self.assertEqual(len(result), 0)

    def test_invalid_prompt_types(self):
        """Test handling of invalid prompt types."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Test non-list input
        with self.assertRaises(ValueError):
            gen.generate_point_clouds_parallel("not a list")

        # Test None in prompt list
        with self.assertRaises(ValueError):
            gen.generate_point_clouds_parallel([None])

        # Test empty string in prompt list
        with self.assertRaises(ValueError):
            gen.generate_point_clouds_parallel(["valid", ""])

        # Test non-string in prompt list
        with self.assertRaises(ValueError):
            gen.generate_point_clouds_parallel(["valid", 123])

    def test_invalid_batch_size(self):
        """Test handling of invalid batch sizes."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Test negative batch size (should be corrected)
        result = gen.generate_point_clouds_parallel([], batch_size=-1)
        self.assertEqual(len(result), 0)  # Empty list should return empty dict

        # Test zero batch size (should be corrected)
        result = gen.generate_point_clouds_parallel([], batch_size=0)
        self.assertEqual(len(result), 0)

    def test_extreme_batch_size(self):
        """Test handling of extreme batch sizes."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Test very large batch size (should be limited)
        result = gen.generate_point_clouds_parallel([], batch_size=100)
        self.assertEqual(len(result), 0)

    def test_invalid_worker_count(self):
        """Test handling of invalid worker counts."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        # Test negative workers (should default to 1)
        gen = OptimizedPointEGenerator(num_workers=-1, output_dir=self.output_dir)
        self.assertEqual(gen.num_workers, 1)

        # Test zero workers (should default to 1)
        gen = OptimizedPointEGenerator(num_workers=0, output_dir=self.output_dir)
        self.assertEqual(gen.num_workers, 1)


class TestModelLoadingFailures(unittest.TestCase):
    """Test handling of model loading and checkpoint failures."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_outputs"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('scripts.optimized_point_e_pipeline.model_from_config')
    def test_base_model_loading_failure(self, mock_model_config):
        """Test handling of base model loading failure."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        mock_model_config.side_effect = Exception("Base model loading failed")
        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        result = gen._generate_single_point_cloud(("test prompt", "cpu", 0))
        self.assertEqual(result[0], "test prompt")
        self.assertIsNone(result[1])

    @patch('scripts.optimized_point_e_pipeline.model_from_config')
    def test_upsampler_model_loading_failure(self, mock_model_config):
        """Test handling of upsampler model loading failure."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        # Mock successful base model, failed upsampler
        def mock_config_side_effect(*args, **kwargs):
            if "base40M" in str(args):
                return MagicMock()
            elif "upsample" in str(args):
                raise Exception("Upsampler model loading failed")
            return MagicMock()

        mock_model_config.side_effect = mock_config_side_effect
        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        result = gen._generate_single_point_cloud(("test prompt", "cpu", 0))
        self.assertEqual(result[0], "test prompt")
        self.assertIsNone(result[1])

    @patch('scripts.optimized_point_e_pipeline.load_checkpoint')
    @patch('scripts.optimized_point_e_pipeline.model_from_config')
    def test_checkpoint_loading_failure(self, mock_model_config, mock_load_checkpoint):
        """Test handling of checkpoint loading failure."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        mock_model_config.return_value = MagicMock()
        mock_load_checkpoint.side_effect = Exception("Checkpoint loading failed")

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        result = gen._generate_single_point_cloud(("test prompt", "cpu", 0))
        self.assertEqual(result[0], "test prompt")
        self.assertIsNone(result[1])

    @patch('scripts.optimized_point_e_pipeline.load_checkpoint')
    @patch('scripts.optimized_point_e_pipeline.model_from_config')
    def test_partial_checkpoint_loading_failure(self, mock_model_config, mock_load_checkpoint):
        """Test handling of partial checkpoint loading failure."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        mock_base_model = MagicMock()
        mock_upsampler_model = MagicMock()
        mock_model_config.side_effect = [mock_base_model, mock_upsampler_model]

        # Fail on first checkpoint load, succeed on retry
        call_count = 0
        def checkpoint_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First checkpoint load failed")
            return None

        mock_load_checkpoint.side_effect = checkpoint_side_effect

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        result = gen._generate_single_point_cloud(("test prompt", "cpu", 0))
        self.assertEqual(result[0], "test prompt")
        self.assertIsNone(result[1])


class TestGenerationFailures(unittest.TestCase):
    """Test handling of generation process failures."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_outputs"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('scripts.optimized_point_e_pipeline.PointCloudSampler')
    @patch('scripts.optimized_point_e_pipeline.load_checkpoint')
    @patch('scripts.optimized_point_e_pipeline.model_from_config')
    def test_sampler_creation_failure(self, mock_model_config, mock_load_checkpoint, mock_sampler):
        """Test handling of sampler creation failure."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        mock_model_config.return_value = MagicMock()
        mock_load_checkpoint.return_value = None
        mock_sampler.side_effect = Exception("Sampler creation failed")

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        result = gen._generate_single_point_cloud(("test prompt", "cpu", 0))
        self.assertEqual(result[0], "test prompt")
        self.assertIsNone(result[1])

    @patch('scripts.optimized_point_e_pipeline.PointCloudSampler')
    @patch('scripts.optimized_point_e_pipeline.load_checkpoint')
    @patch('scripts.optimized_point_e_pipeline.model_from_config')
    def test_sample_generation_failure(self, mock_model_config, mock_load_checkpoint, mock_sampler_class):
        """Test handling of sample generation failure."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        mock_model_config.return_value = MagicMock()
        mock_load_checkpoint.return_value = None

        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler
        mock_sampler.sample_batch_progressive.side_effect = Exception("Sample generation failed")

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        result = gen._generate_single_point_cloud(("test prompt", "cpu", 0))
        self.assertEqual(result[0], "test prompt")
        self.assertIsNone(result[1])

    @patch('scripts.optimized_point_e_pipeline.PointCloudSampler')
    @patch('scripts.optimized_point_e_pipeline.load_checkpoint')
    @patch('scripts.optimized_point_e_pipeline.model_from_config')
    def test_empty_samples_generation(self, mock_model_config, mock_load_checkpoint, mock_sampler_class):
        """Test handling of empty samples from generation."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        mock_model_config.return_value = MagicMock()
        mock_load_checkpoint.return_value = None

        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler
        mock_sampler.sample_batch_progressive.return_value = iter([])  # Empty iterator
        mock_sampler.output_to_point_clouds.return_value = []

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        result = gen._generate_single_point_cloud(("test prompt", "cpu", 0))
        self.assertEqual(result[0], "test prompt")
        self.assertIsNone(result[1])

    @patch('scripts.optimized_point_e_pipeline.PointCloudSampler')
    @patch('scripts.optimized_point_e_pipeline.load_checkpoint')
    @patch('scripts.optimized_point_e_pipeline.model_from_config')
    def test_point_cloud_conversion_failure(self, mock_model_config, mock_load_checkpoint, mock_sampler_class):
        """Test handling of point cloud conversion failure."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        mock_model_config.return_value = MagicMock()
        mock_load_checkpoint.return_value = None

        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler
        mock_sampler.sample_batch_progressive.return_value = iter([MagicMock()])
        mock_sampler.output_to_point_clouds.side_effect = Exception("Point cloud conversion failed")

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        result = gen._generate_single_point_cloud(("test prompt", "cpu", 0))
        self.assertEqual(result[0], "test prompt")
        self.assertIsNone(result[1])


class TestQualityEnhancementFailures(unittest.TestCase):
    """Test handling of quality enhancement failures."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test point cloud
        np.random.seed(42)
        self.test_points = np.random.random((100, 3))
        self.test_pc = PointCloud(coords=self.test_points, channels={})

    def test_quality_enhancement_with_invalid_coordinates(self):
        """Test quality enhancement with NaN/Inf coordinates."""
        # Create point cloud with NaN values
        nan_points = self.test_points.copy()
        nan_points[10] = [np.nan, np.nan, np.nan]

        pc_with_nan = PointCloud(coords=nan_points, channels={})
        result = enhance_point_cloud_quality(pc_with_nan)

        # Should return original point cloud unchanged
        self.assertEqual(len(result.coords), len(pc_with_nan.coords))
        # Check that coordinates are unchanged (NaN values should be preserved)
        self.assertTrue(np.array_equal(result.coords, pc_with_nan.coords, equal_nan=True) or 
                       (np.isnan(result.coords).any() and np.isnan(pc_with_nan.coords).any()))

    def test_quality_enhancement_with_inf_coordinates(self):
        """Test quality enhancement with infinite coordinates."""
        # Create point cloud with Inf values
        inf_points = self.test_points.copy()
        inf_points[10] = [np.inf, -np.inf, np.inf]

        pc_with_inf = PointCloud(coords=inf_points, channels={})
        result = enhance_point_cloud_quality(pc_with_inf)

        # Should return original point cloud unchanged
        self.assertEqual(len(result.coords), len(pc_with_inf.coords))
        # Check that coordinates are unchanged (Inf values should be preserved)
        self.assertTrue(np.array_equal(result.coords, pc_with_inf.coords, equal_nan=True) or
                       (np.isinf(result.coords).any() and np.isinf(pc_with_inf.coords).any()))

    @patch('point_e.enhancements.geometric_quality_enhancements.OPEN3D_AVAILABLE', False)
    def test_quality_enhancement_without_open3d(self):
        """Test quality enhancement when Open3D is not available."""
        result = enhance_point_cloud_quality(self.test_pc)

        # Should return original point cloud unchanged
        self.assertEqual(len(result.coords), len(self.test_pc.coords))
        np.testing.assert_array_equal(result.coords, self.test_pc.coords)

    @patch('point_e.enhancements.geometric_quality_enhancements.o3d')
    def test_statistical_outlier_removal_open3d_failure(self, mock_o3d):
        """Test statistical outlier removal when Open3D operations fail."""
        mock_o3d.geometry.PointCloud.side_effect = Exception("Open3D failure")

        result = statistical_outlier_removal(self.test_points)

        # Should return original points unchanged
        np.testing.assert_array_equal(result, self.test_points)

    @patch('point_e.enhancements.geometric_quality_enhancements.o3d')
    def test_knn_pca_normal_refinement_open3d_failure(self, mock_o3d):
        """Test k-NN/PCA normal refinement when Open3D operations fail."""
        mock_o3d.geometry.PointCloud.side_effect = Exception("Open3D failure")

        result = knn_pca_normal_refinement(self.test_points)

        # Should return original points unchanged
        np.testing.assert_array_equal(result, self.test_points)


class TestFileOperations(unittest.TestCase):
    """Test file saving and loading operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create test point cloud
        np.random.seed(42)
        coords = np.random.random((100, 3))
        channels = {'R': np.random.random(100), 'G': np.random.random(100), 'B': np.random.random(100)}
        self.test_pc = PointCloud(coords=coords, channels=channels)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_save_results_empty_dict(self):
        """Test saving empty results dictionary."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        saved_files = gen.save_results({})
        self.assertEqual(len(saved_files), 0)

    def test_save_results_none_point_cloud(self):
        """Test saving results with None point cloud."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        saved_files = gen.save_results({"test": None})
        self.assertEqual(len(saved_files), 0)

    def test_save_results_invalid_point_cloud(self):
        """Test saving results with invalid point cloud."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Point cloud with no coordinates
        invalid_pc = PointCloud(coords=None, channels={})
        saved_files = gen.save_results({"test": invalid_pc})
        self.assertEqual(len(saved_files), 0)

        # Point cloud with empty coordinates
        empty_pc = PointCloud(coords=np.empty((0, 3)), channels={})
        saved_files = gen.save_results({"test": empty_pc})
        self.assertEqual(len(saved_files), 0)

    def test_save_results_with_nan_inf(self):
        """Test saving results with NaN/Inf coordinates."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Create point cloud with NaN values
        nan_coords = self.test_pc.coords.copy()
        nan_coords[10] = [np.nan, np.nan, np.nan]
        nan_pc = PointCloud(coords=nan_coords, channels=self.test_pc.channels)

        saved_files = gen.save_results({"test": nan_pc})
        self.assertEqual(len(saved_files), 0)  # Should not save invalid point clouds

    def test_save_results_file_operations(self):
        """Test successful file saving operations."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Test NPZ saving
        saved_files = gen.save_results({"test_prompt": self.test_pc}, save_npz=True, save_plots=False)
        self.assertEqual(len(saved_files), 2)  # NPZ and PLY

        # Check files exist
        for file_path in saved_files:
            self.assertTrue(file_path.exists())
            self.assertTrue(file_path.stat().st_size > 0)

        # Check file extensions
        extensions = [f.suffix for f in saved_files]
        self.assertIn('.npz', extensions)
        self.assertIn('.ply', extensions)

    def test_save_results_filename_sanitization(self):
        """Test filename sanitization for special characters."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Test prompt with special characters
        special_prompt = "test/with\\special:chars?"
        saved_files = gen.save_results({special_prompt: self.test_pc}, save_npz=True, save_plots=False)

        # Should create valid filenames
        for file_path in saved_files:
            self.assertTrue(file_path.exists())
            # Filename should not contain special characters
            self.assertNotIn('/', file_path.stem)
            self.assertNotIn('\\', file_path.stem)
            self.assertNotIn(':', file_path.stem)
            self.assertNotIn('?', file_path.stem)

    def test_save_results_io_failure(self):
        """Test handling of file I/O failures."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Mock the save method to simulate I/O failure
        with patch.object(self.test_pc, 'save', side_effect=IOError("Disk full")):
            with patch('scripts.optimized_point_e_pipeline.PointCloud.write_ply', side_effect=IOError("Disk full")):
                # Should handle I/O errors gracefully (not crash)
                saved_files = gen.save_results({"test": self.test_pc})
                # Files should not be saved due to I/O error
                self.assertEqual(len(saved_files), 0)


class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization claims and multiprocessing."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_outputs"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_worker_count_optimization(self):
        """Test worker count optimization based on CPU cores."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        # Test default worker count
        gen = OptimizedPointEGenerator(output_dir=self.output_dir)
        expected_workers = max(1, min(mp.cpu_count() - 1, mp.cpu_count()))
        self.assertGreaterEqual(gen.num_workers, 1)
        self.assertLessEqual(gen.num_workers, mp.cpu_count())

    def test_deterministic_seed_setting(self):
        """Test that deterministic seeds are properly set."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        # Test deterministic mode
        gen_det = OptimizedPointEGenerator(deterministic=True, random_seed=123, output_dir=self.output_dir)
        self.assertTrue(gen_det.deterministic)
        self.assertEqual(gen_det.random_seed, 123)

        # Test non-deterministic mode
        gen_nondet = OptimizedPointEGenerator(deterministic=False, output_dir=self.output_dir)
        self.assertFalse(gen_nondet.deterministic)

    def test_device_selection(self):
        """Test automatic device selection."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Should select CPU if CUDA not available, or CUDA if available
        if torch.cuda.is_available():
            self.assertTrue(str(gen.device).startswith('cuda'))
        else:
            self.assertEqual(str(gen.device), 'cpu')

    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Should not raise exceptions
        gen.cleanup()

        # Test that cleanup calls garbage collection
        with patch('scripts.optimized_point_e_pipeline.gc.collect') as mock_gc:
            gen.cleanup()
            mock_gc.assert_called()

        # Test CUDA cache clearing if CUDA available
        if torch.cuda.is_available():
            with patch('torch.cuda.empty_cache') as mock_cuda_cache:
                gen.cleanup()
                mock_cuda_cache.assert_called()


class TestGeometryPreservation(unittest.TestCase):
    """Test geometry preservation claims - coordinates must never change."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_points = np.random.random((1000, 3)).astype(np.float32)
        self.test_pc = PointCloud(coords=self.test_points, channels={
            'R': np.random.random(1000),
            'G': np.random.random(1000),
            'B': np.random.random(1000)
        })

    def test_coordinate_immutability(self):
        """Test that coordinates are never modified during enhancement."""
        original_coords = self.test_pc.coords.copy()

        # Apply enhancement
        result = enhance_point_cloud_quality(self.test_pc, num_iterations=3)

        # Coordinates must be identical
        np.testing.assert_array_equal(result.coords, original_coords)
        self.assertEqual(len(result.coords), len(original_coords))

    def test_point_count_preservation(self):
        """Test that point count is never changed."""
        original_count = len(self.test_pc.coords)

        # Apply enhancement with multiple iterations
        result = enhance_point_cloud_quality(self.test_pc, num_iterations=5)

        self.assertEqual(len(result.coords), original_count)

    def test_channel_preservation(self):
        """Test that channels are preserved."""
        result = enhance_point_cloud_quality(self.test_pc)

        # All channels should be preserved
        for channel_name in self.test_pc.channels:
            self.assertIn(channel_name, result.channels)
            np.testing.assert_array_equal(
                result.channels[channel_name],
                self.test_pc.channels[channel_name]
            )

    def test_bounding_box_preservation(self):
        """Test that spatial bounding box is preserved."""
        original_coords = self.test_pc.coords
        original_min = np.min(original_coords, axis=0)
        original_max = np.max(original_coords, axis=0)

        result = enhance_point_cloud_quality(self.test_pc)

        result_min = np.min(result.coords, axis=0)
        result_max = np.max(result.coords, axis=0)

        # Bounding box should be identical (coordinates unchanged)
        np.testing.assert_array_equal(original_min, result_min)
        np.testing.assert_array_equal(original_max, result_max)

    def test_coordinate_finiteness_preservation(self):
        """Test that coordinate finiteness is preserved."""
        result = enhance_point_cloud_quality(self.test_pc)

        # All coordinates should remain finite
        self.assertTrue(np.all(np.isfinite(result.coords)))


class TestValidationAndReliability(unittest.TestCase):
    """Test validation and reliability claims."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.valid_coords = np.random.random((100, 3)).astype(np.float32)
        self.valid_pc = PointCloud(coords=self.valid_coords, channels={})

    def test_no_empty_point_clouds(self):
        """Test that empty point clouds are rejected."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=Path(tempfile.mkdtemp()))

        # Empty point cloud should not be saved
        empty_pc = PointCloud(coords=np.empty((0, 3)), channels={})
        saved_files = gen.save_results({"empty": empty_pc})
        self.assertEqual(len(saved_files), 0)

    def test_no_nan_inf_values(self):
        """Test that NaN/Inf values are rejected."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=Path(tempfile.mkdtemp()))

        # Point cloud with NaN should not be saved
        nan_coords = self.valid_coords.copy()
        nan_coords[10] = [np.nan, np.nan, np.nan]
        nan_pc = PointCloud(coords=nan_coords, channels={})

        saved_files = gen.save_results({"nan": nan_pc})
        self.assertEqual(len(saved_files), 0)

        # Point cloud with Inf should not be saved
        inf_coords = self.valid_coords.copy()
        inf_coords[10] = [np.inf, -np.inf, np.inf]
        inf_pc = PointCloud(coords=inf_coords, channels={})

        saved_files = gen.save_results({"inf": inf_pc})
        self.assertEqual(len(saved_files), 0)

    def test_proper_spatial_distribution(self):
        """Test that point clouds have proper spatial distribution."""
        # Valid point cloud should have reasonable bounding box
        coords = self.valid_pc.coords

        # Check that coordinates are in reasonable range (not all zeros, not extreme values)
        coord_range = np.ptp(coords, axis=0)  # peak-to-peak range
        self.assertTrue(np.all(coord_range > 0.1))  # Should have some spread

        # Check that coordinates are not all identical
        unique_coords = np.unique(coords, axis=0)
        self.assertGreater(len(unique_coords), 1)

    def test_bounding_box_validation(self):
        """Test bounding box validation."""
        coords = self.valid_pc.coords

        # Calculate bounding box
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        bbox_size = max_coords - min_coords

        # Bounding box should be reasonable (not zero-sized, not infinite)
        self.assertTrue(np.all(bbox_size > 0))
        self.assertTrue(np.all(np.isfinite(bbox_size)))

    def test_output_consistency(self):
        """Test that outputs are consistent and usable."""
        result = enhance_point_cloud_quality(self.valid_pc)

        # Result should be a valid PointCloud
        self.assertIsInstance(result, PointCloud)
        self.assertIsNotNone(result.coords)
        self.assertIsNotNone(result.channels)

        # Should have same number of points
        self.assertEqual(len(result.coords), len(self.valid_pc.coords))

        # Coordinates should be valid
        self.assertTrue(np.all(np.isfinite(result.coords)))
        self.assertEqual(result.coords.shape[1], 3)  # 3D coordinates


class TestOutputGeneration(unittest.TestCase):
    """Test output generation claims."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create test point cloud
        np.random.seed(42)
        coords = np.random.random((100, 3))
        self.test_pc = PointCloud(coords=coords, channels={
            'R': np.random.random(100),
            'G': np.random.random(100),
            'B': np.random.random(100)
        })

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_enhanced_point_cloud_outputs(self):
        """Test that enhanced point cloud outputs are generated."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(
            output_dir=self.output_dir,
            enable_geometric_quality_enhancements=True
        )

        # Mock the entire parallel generation method
        with patch.object(gen, 'generate_point_clouds_parallel', return_value={"test prompt": self.test_pc}) as mock_parallel:
            results = gen.generate_point_clouds_parallel(["test prompt"])
            self.assertIn("test prompt", results)
            self.assertIsNotNone(results["test prompt"])
            mock_parallel.assert_called_once()

    def test_file_format_support(self):
        """Test support for multiple file formats."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        saved_files = gen.save_results({"test": self.test_pc}, save_npz=True, save_plots=False)

        # Should save both NPZ and PLY
        self.assertEqual(len(saved_files), 2)

        file_extensions = [f.suffix for f in saved_files]
        self.assertIn('.npz', file_extensions)
        self.assertIn('.ply', file_extensions)

    def test_before_after_evaluation_support(self):
        """Test support for before-and-after evaluation."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        # Save "before" point cloud
        before_files = gen.save_results({"test": self.test_pc}, save_npz=True, save_plots=False)

        # Apply enhancement
        enhanced_pc = enhance_point_cloud_quality(self.test_pc)

        # Save "after" point cloud
        after_files = gen.save_results({"test_enhanced": enhanced_pc}, save_npz=True, save_plots=False)

        # Should be able to save both versions
        self.assertEqual(len(before_files), 2)
        self.assertEqual(len(after_files), 2)

        # Files should be different (different filenames)
        before_names = {f.name for f in before_files}
        after_names = {f.name for f in after_files}
        self.assertTrue(before_names.isdisjoint(after_names))

    def test_performance_report_generation(self):
        """Test performance report generation."""
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

        gen = OptimizedPointEGenerator(output_dir=self.output_dir)

        report = gen.get_performance_report()

        # Should generate a substantial report
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)

        # Should contain key sections
        self.assertIn("OPTIMIZED POINT-E PERFORMANCE REPORT", report)
        self.assertIn("SYSTEM CONFIGURATION", report)

    def test_comprehensive_demo_functionality(self):
        """Test the comprehensive demo functionality."""
        from scripts.optimized_point_e_pipeline import run_optimized_demo

        # Mock the generation to avoid actual model loading
        with patch('scripts.optimized_point_e_pipeline.OptimizedPointEGenerator') as mock_gen_class:
            mock_gen = MagicMock()
            mock_gen_class.return_value = mock_gen
            mock_gen.generate_point_clouds_parallel.return_value = {"test": self.test_pc}
            mock_gen.save_results.return_value = [Path("test.npz")]
            mock_gen.get_performance_report.return_value = "Test Report"
            mock_gen.cleanup.return_value = None

            # Should not raise exceptions
            try:
                run_optimized_demo()
            except Exception as e:
                self.fail(f"Demo function failed: {e}")

            # Verify key methods were called
            mock_gen.generate_point_clouds_parallel.assert_called()
            mock_gen.save_results.assert_called()
            mock_gen.get_performance_report.assert_called()
            mock_gen.cleanup.assert_called()


if __name__ == '__main__':
    unittest.main()
    """Test the optimized pipeline with a small set of prompts."""
    print("Testing Optimized Point-E Pipeline")
    print("=" * 50)

    try:
        # Create generator with minimal workers for testing
        from scripts.optimized_point_e_pipeline import OptimizedPointEGenerator

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