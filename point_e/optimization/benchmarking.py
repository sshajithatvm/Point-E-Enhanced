"""
Performance benchmarking utilities for Point-E.
Measures and compares inference speed, quality metrics, and memory usage.
"""

import time
import torch
import numpy as np
import psutil
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime

from .logger import get_logger, MetricsLogger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    num_samples: int
    total_time: float  # seconds
    time_per_sample: float  # seconds
    throughput: float  # samples/second
    memory_peak_mb: float
    memory_avg_mb: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"Benchmark: {self.name}\n"
            f"  Samples: {self.num_samples}\n"
            f"  Total time: {self.total_time:.2f}s\n"
            f"  Per sample: {self.time_per_sample:.3f}s ({self.throughput:.2f} samples/s)\n"
            f"  Memory: peak={self.memory_peak_mb:.1f}MB, avg={self.memory_avg_mb:.1f}MB"
        )


class ResourceMonitor:
    """Monitors CPU/GPU resources during execution."""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.monitoring = False
    
    def start(self) -> None:
        """Start monitoring resources."""
        self.memory_samples.clear()
        self.cpu_samples.clear()
        self.monitoring = True
        self._record_sample()
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        
        if not self.memory_samples:
            return {
                "memory_peak_mb": 0,
                "memory_avg_mb": 0,
                "cpu_avg_percent": 0,
            }
        
        return {
            "memory_peak_mb": max(self.memory_samples),
            "memory_avg_mb": np.mean(self.memory_samples),
            "cpu_avg_percent": np.mean(self.cpu_samples) if self.cpu_samples else 0,
        }
    
    def _record_sample(self) -> None:
        """Record current resource usage."""
        try:
            # Memory in MB
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.01)
            self.cpu_samples.append(cpu_percent)
        except Exception as e:
            logger.warning(f"Failed to record resource sample: {e}")


class PerformanceBenchmark:
    """
    Main benchmarking utility for Point-E inference.
    Measures end-to-end performance and quality metrics.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_logger = MetricsLogger()
        self.results: List[BenchmarkResult] = []
    
    def benchmark_inference(
        self,
        inference_fn,
        num_samples: int = 10,
        name: str = "inference",
        **fn_kwargs,
    ) -> BenchmarkResult:
        """
        Benchmark an inference function.
        
        Args:
            inference_fn: Function to benchmark. Should accept **fn_kwargs
            num_samples: Number of samples to process
            name: Benchmark name
            **fn_kwargs: Arguments to pass to inference_fn
        
        Returns:
            BenchmarkResult with timing and resource metrics
        """
        monitor = ResourceMonitor()
        
        logger.info(f"Starting benchmark: {name} ({num_samples} samples)")
        
        # Warmup
        try:
            inference_fn(**fn_kwargs)
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        
        # Actual benchmark
        monitor.start()
        start_time = time.time()
        
        for i in range(num_samples):
            try:
                inference_fn(**fn_kwargs)
            except Exception as e:
                logger.error(f"Inference failed at sample {i}: {e}")
                num_samples = i  # Adjust count
                break
        
        total_time = time.time() - start_time
        resource_stats = monitor.stop()
        
        # Create result
        result = BenchmarkResult(
            name=name,
            num_samples=num_samples,
            total_time=total_time,
            time_per_sample=total_time / num_samples if num_samples > 0 else 0,
            throughput=num_samples / total_time if total_time > 0 else 0,
            memory_peak_mb=resource_stats["memory_peak_mb"],
            memory_avg_mb=resource_stats["memory_avg_mb"],
            timestamp=datetime.utcnow().isoformat(),
        )
        
        self.results.append(result)
        logger.info(str(result))
        
        return result
    
    def compare_results(
        self,
        baseline: BenchmarkResult,
        optimized: BenchmarkResult,
    ) -> Dict[str, float]:
        """
        Compare baseline and optimized results.
        
        Returns:
            Dictionary with speedup factors and improvements
        """
        speedup = baseline.time_per_sample / optimized.time_per_sample if optimized.time_per_sample > 0 else 1
        throughput_improvement = (optimized.throughput - baseline.throughput) / baseline.throughput * 100 if baseline.throughput > 0 else 0
        memory_reduction = (baseline.memory_peak_mb - optimized.memory_peak_mb) / baseline.memory_peak_mb * 100 if baseline.memory_peak_mb > 0 else 0
        
        comparison = {
            "speedup_factor": round(speedup, 2),
            "throughput_improvement_percent": round(throughput_improvement, 1),
            "memory_reduction_percent": round(memory_reduction, 1),
            "baseline_time_per_sample": round(baseline.time_per_sample, 3),
            "optimized_time_per_sample": round(optimized.time_per_sample, 3),
            "baseline_memory_mb": round(baseline.memory_peak_mb, 1),
            "optimized_memory_mb": round(optimized.memory_peak_mb, 1),
        }
        
        return comparison
    
    def generate_report(self) -> str:
        """Generate a text report of all benchmarks."""
        report = "=" * 70 + "\n"
        report += "PERFORMANCE BENCHMARK REPORT\n"
        report += "=" * 70 + "\n\n"
        
        for result in self.results:
            report += str(result) + "\n\n"
        
        # Comparisons
        if len(self.results) >= 2:
            report += "=" * 70 + "\n"
            report += "COMPARISONS\n"
            report += "=" * 70 + "\n\n"
            
            for i in range(1, len(self.results)):
                baseline = self.results[0]
                optimized = self.results[i]
                comparison = self.compare_results(baseline, optimized)
                
                report += f"Speedup ({baseline.name} → {optimized.name}):\n"
                report += f"  Speedup factor: {comparison['speedup_factor']:.2f}x\n"
                report += f"  Throughput improvement: {comparison['throughput_improvement_percent']:.1f}%\n"
                report += f"  Memory reduction: {comparison['memory_reduction_percent']:.1f}%\n\n"
        
        return report
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save benchmark results to JSON file.
        
        Args:
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "num_benchmarks": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved benchmark results to {filepath}")
        return filepath


def create_benchmark_suite(
    base_inference_fn,
    optimized_inference_fn,
    num_samples: int = 5,
) -> Tuple[BenchmarkResult, BenchmarkResult, Dict[str, float]]:
    """
    Create a benchmark comparing baseline vs optimized inference.
    
    Args:
        base_inference_fn: Baseline inference function
        optimized_inference_fn: Optimized inference function  
        num_samples: Number of samples for each benchmark
    
    Returns:
        Tuple of (baseline_result, optimized_result, comparison_dict)
    """
    benchmarker = PerformanceBenchmark()
    
    logger.info("=" * 70)
    logger.info("RUNNING BENCHMARK SUITE")
    logger.info("=" * 70)
    
    # Baseline benchmark
    baseline = benchmarker.benchmark_inference(
        base_inference_fn,
        num_samples=num_samples,
        name="baseline",
    )
    
    # Optimized benchmark
    optimized = benchmarker.benchmark_inference(
        optimized_inference_fn,
        num_samples=num_samples,
        name="optimized",
    )
    
    # Generate comparison
    comparison = benchmarker.compare_results(baseline, optimized)
    
    # Print report
    report = benchmarker.generate_report()
    logger.info("\n" + report)
    
    # Save results
    benchmarker.save_results()
    
    return baseline, optimized, comparison
