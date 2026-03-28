"""
Optimized Point-E pipeline with multiprocessing and advanced geometric quality enhancements.
Features:
- Multiprocessing for parallel point cloud generation
- Advanced batching with memory management
- Geometric quality enhancements with edge-aware refinement and adaptive density
- Stable results with comprehensive error handling
- End-to-end pipeline with file saving
"""

import torch
import numpy as np
import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import gc

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
from point_e.enhancements.geometric_quality_enhancements import enhance_point_cloud_quality
from point_e.optimization.performance_optimizer import (
    PerformanceOptimizer,
    OptimizationConfig,
)
from point_e.optimization.logger import (
    get_logger,
    get_metrics_logger,
    initialize_logging,
    log_performance,
)

logger = get_logger(__name__)
metrics_logger = get_metrics_logger(__name__)


class OptimizedPointEGenerator:
    """
    Optimized Point-E generator with multiprocessing and geometric quality enhancements.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        optimizer_config: Optional[OptimizationConfig] = None,
        output_dir: Path = Path("optimized_point_cloud_outputs"),
        num_workers: Optional[int] = None,
        enable_geometric_quality_enhancements: bool = True,
        deterministic: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize the optimized generator.

        Args:
            device: Torch device (default: auto-detect)
            optimizer_config: Performance optimizer configuration
            output_dir: Directory for saving outputs
            num_workers: Number of parallel workers (default: CPU count - 1)
            enable_geometric_quality_enhancements: Use local neighborhood quality enhancements
            deterministic: Enable deterministic outputs for reproducibility
            random_seed: Random seed for deterministic mode
        """
        # Setup device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Setup multiprocessing
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        self.num_workers = num_workers

        # Setup deterministic behavior
        self.deterministic = deterministic
        self.random_seed = random_seed
        if deterministic:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            logger.info(f"Deterministic mode enabled with seed: {random_seed}")
        else:
            logger.info("Deterministic mode disabled")

        # Setup optimizer
        if optimizer_config is None:
            optimizer_config = OptimizationConfig(
                device=str(device),
                batch_size=2,  # Smaller batches for stability
                num_workers=num_workers,
            )
        self.optimizer = PerformanceOptimizer(optimizer_config)
        self.config = optimizer_config

        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Enhancement settings
        self.enable_geometric_quality_enhancements = enable_geometric_quality_enhancements

        logger.info("OptimizedPointEGenerator initialized")

    def _generate_single_point_cloud(self, args) -> Tuple[str, Optional[PointCloud]]:
        """
        Generate a single point cloud for a prompt (worker function).
        Includes retry logic and robust error handling.
        Args should contain: (prompt, device_str, retry_count)
        """
        prompt, device_str, retry_count = args

        try:
            # Setup device in worker
            device = torch.device(device_str)

            # Set deterministic seed for this worker if enabled
            if self.deterministic:
                worker_seed = self.random_seed + hash(prompt) % 1000  # Deterministic per prompt
                torch.manual_seed(worker_seed)
                np.random.seed(worker_seed)

            # Load models in worker process with error handling
            try:
                base_model = model_from_config(MODEL_CONFIGS["base40M-textvec"], device)
                base_model.eval()
                base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["base40M-textvec"])
            except Exception as e:
                logger.error(f"Failed to load base model: {e}")
                raise

            try:
                upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
                upsampler_model.eval()
                upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])
            except Exception as e:
                logger.error(f"Failed to load upsampler model: {e}")
                raise

            # Load checkpoints with retry
            max_checkpoint_retries = 3
            for attempt in range(max_checkpoint_retries):
                try:
                    base_model.load_state_dict(load_checkpoint("base40M-textvec", device))
                    upsampler_model.load_state_dict(load_checkpoint("upsample", device))
                    break
                except Exception as e:
                    if attempt == max_checkpoint_retries - 1:
                        logger.error(f"Failed to load checkpoints after {max_checkpoint_retries} attempts: {e}")
                        raise
                    logger.warning(f"Checkpoint loading attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(1)

            # Create sampler with optimized parameters
            sampler = PointCloudSampler(
                device=device,
                models=[base_model, upsampler_model],
                diffusions=[base_diffusion, upsampler_diffusion],
                num_points=[1024, 4096 - 1024],
                aux_channels=["R", "G", "B"],
                guidance_scale=[3.0, 0.0],
                model_kwargs_key_filter=("texts", ""),
                karras_steps=(48, 48),  # Optimized steps
            )

            # Generate samples with timeout and error handling
            samples = None
            generation_start = time.time()

            try:
                for x in sampler.sample_batch_progressive(
                    batch_size=1,
                    model_kwargs=dict(texts=[prompt]),
                ):
                    samples = x
                    # Check for timeout (5 minutes per prompt)
                    if time.time() - generation_start > 300:
                        raise TimeoutError(f"Generation timeout for prompt: {prompt}")

            except Exception as e:
                logger.error(f"Sample generation failed for '{prompt}': {e}")
                if retry_count < 2:  # Allow up to 2 retries
                    logger.info(f"Retrying generation for '{prompt}' (attempt {retry_count + 1})")
                    return self._generate_single_point_cloud((prompt, device_str, retry_count + 1))
                raise

            if samples is None:
                logger.error(f"No samples generated for '{prompt}' after all attempts")
                return prompt, None

            # Convert to point cloud
            try:
                pc = sampler.output_to_point_clouds(samples)[0]
                if pc is None or len(pc.coords) == 0:
                    raise ValueError("Empty point cloud generated")
            except Exception as e:
                logger.error(f"Point cloud conversion failed for '{prompt}': {e}")
                return prompt, None

            # Apply geometric quality enhancements if enabled
            if self.enable_geometric_quality_enhancements:
                try:
                    pc = enhance_point_cloud_quality(pc, num_iterations=1)
                except Exception as e:
                    logger.warning(f"Quality enhancement failed for '{prompt}', using original: {e}")

            generation_time = time.time() - generation_start
            logger.info(f"Successfully generated point cloud for '{prompt}' in {generation_time:.2f}s")
            return prompt, pc

        except Exception as e:
            logger.error(f"Failed to generate point cloud for '{prompt}' after all retries: {e}")
            return prompt, None

    def generate_point_clouds_parallel(
        self,
        prompts: List[str],
        batch_size: int = 1,
    ) -> Dict[str, PointCloud]:
        """
        Generate multiple point clouds in parallel using multiprocessing.

        Args:
            prompts: List of text prompts
            batch_size: Number of samples per prompt (kept at 1 for stability)

        Returns:
            Dictionary mapping prompts to PointCloud objects
        """
        results = {}
        total_prompts = len(prompts)

        logger.info(f"Generating {total_prompts} point clouds using {self.num_workers} workers")

        with log_performance("parallel_generation", context={"num_prompts": total_prompts}):
            # Prepare arguments for each worker
            worker_args = [
                (prompt, str(self.device), 0)  # (prompt, device_str, retry_count)
                for prompt in prompts
            ]

            # Use ProcessPoolExecutor for parallel generation
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all generation tasks
                future_to_prompt = {
                    executor.submit(self._generate_single_point_cloud, args): args[0]
                    for args in worker_args
                }

                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_prompt):
                    prompt = future_to_prompt[future]
                    try:
                        result_prompt, pc = future.result()
                        if pc is not None:
                            results[result_prompt] = pc
                            logger.info(f"✓ Generated: '{result_prompt}' ({len(pc.coords)} points)")
                        else:
                            logger.error(f"✗ Failed: '{result_prompt}'")
                    except Exception as e:
                        logger.error(f"✗ Exception for '{prompt}': {e}")

                    completed += 1
                    if completed % 5 == 0 or completed == total_prompts:
                        logger.info(f"Progress: {completed}/{total_prompts} point clouds generated")

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Parallel generation complete: {len(results)}/{total_prompts} successful")
        return results

    def save_results(
        self,
        point_clouds: Dict[str, PointCloud],
        save_plots: bool = True,
        save_npz: bool = True,
    ) -> List[Path]:
        """
        Save generated point clouds to disk with comprehensive error handling.

        Args:
            point_clouds: Dictionary mapping prompts to PointClouds
            save_plots: Whether to save visualization plots
            save_npz: Whether to save NPZ files

        Returns:
            List of saved file paths
        """
        saved_files = []

        for prompt, pc in point_clouds.items():
            try:
                # Sanitize filename
                safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in prompt)
                safe_name = safe_name.replace(" ", "_")[:50]

                base_path = self.output_dir / safe_name

                # Save NPZ if requested
                if save_npz:
                    npz_path = base_path.with_suffix('.npz')
                    pc.save(npz_path)
                    saved_files.append(npz_path)
                    logger.info(f"Saved NPZ: {npz_path}")

                # Save PLY format for compatibility
                ply_path = base_path.with_suffix('.ply')
                pc.save(ply_path)
                saved_files.append(ply_path)
                logger.info(f"Saved PLY: {ply_path}")

                # Save plot if requested
                if save_plots:
                    try:
                        plot_path = base_path.with_suffix('.png')
                        fig = plot_point_cloud(pc)
                        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
                        fig.close()  # Important: close to free memory
                        saved_files.append(plot_path)
                        logger.info(f"Saved plot: {plot_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save plot for '{prompt}': {e}")

            except Exception as e:
                logger.error(f"Failed to save results for '{prompt}': {e}")

        return saved_files

    def get_performance_report(self) -> str:
        """Generate comprehensive performance analysis report."""
        report = "=" * 80 + "\n"
        report += "OPTIMIZED POINT-E PERFORMANCE REPORT\n"
        report += "=" * 80 + "\n\n"

        # Timing stats
        timing_stats = self.optimizer.get_timing_stats()
        if timing_stats:
            report += "TIMING STATISTICS\n"
            report += "-" * 80 + "\n"
            for stage, stats in timing_stats.items():
                report += f"{stage}:\n"
                report += f"  Total: {stats['total']:.2f}s\n"
                report += f"  Mean: {stats['mean']:.3f}s ± {stats['std']:.3f}s\n"
                report += f"  Range: {stats['min']:.3f}s - {stats['max']:.3f}s\n"
                report += f"  Count: {stats['count']}\n"
            report += "\n"

        # Cache stats
        cache_stats = self.optimizer.get_cache_stats()
        if cache_stats:
            report += "CACHE STATISTICS\n"
            report += "-" * 80 + "\n"
            report += f"Hits: {cache_stats['cache_hits']}\n"
            report += f"Misses: {cache_stats['cache_misses']}\n"
            report += f"Hit rate: {cache_stats['hit_rate']:.1%}\n"
            report += f"Cache size: {cache_stats['cache_size']} entries\n"
            report += "\n"

        # System info
        report += "SYSTEM CONFIGURATION\n"
        report += "-" * 80 + "\n"
        report += f"Device: {self.device}\n"
        report += f"Workers: {self.num_workers}\n"
        report += f"Geometric quality enhancements: {self.enable_geometric_quality_enhancements}\n"
        report += f"Batch size: {self.config.batch_size}\n"

        return report

    def cleanup(self):
        """Clean up resources."""
        # Clear caches
        self.optimizer.clear_caches()

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Resources cleaned up")


def run_optimized_demo():
    """
    Demonstration of optimized Point-E pipeline with multiprocessing and geometric quality enhancements.
    """
    # Initialize logging
    log_dir = Path("optimized_point_e_logs")
    initialize_logging(log_dir=log_dir)

    logger.info("Starting Optimized Point-E Demo")
    logger.info("=" * 80)

    try:
        # Create optimized generator
        gen = OptimizedPointEGenerator(
            output_dir=Path("optimized_point_cloud_outputs"),
            enable_geometric_quality_enhancements=True,
        )

        # Test prompts
        prompts = [
            "a red motorcycle",
            "a blue cube",
            "a purple sphere",
        ]

        logger.info(f"Generating {len(prompts)} point clouds with optimizations...")

        # Generate in parallel
        results = gen.generate_point_clouds_parallel(prompts, batch_size=1)

        # Save results
        logger.info("Saving results...")
        saved_files = gen.save_results(results, save_plots=True, save_npz=True)
        logger.info(f"Saved {len(saved_files)} files")

        # Print performance report
        report = gen.get_performance_report()
        logger.info("\n" + report)

        # Save performance report
        report_path = gen.output_dir / "performance_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Performance report saved to: {report_path}")

        logger.info("✓ Optimized demo completed successfully")

    except Exception as e:
        logger.error(f"Optimized demo failed: {e}", exc_info=True)
        raise
    finally:
        gen.cleanup()


if __name__ == "__main__":
    run_optimized_demo()