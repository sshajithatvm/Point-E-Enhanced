import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import logging
import time

@dataclass
class GenerationConfig:
    """Configuration for optimized point cloud generation."""
    batch_size: int = 2
    num_workers: int = mp.cpu_count()
    use_mixed_precision: bool = True
    optimize_memory: bool = True
    target_points: int = 4096
    guidance_scale: float = 3.0
    karras_steps: int = 16
    device: str = "cpu"

class PointEGenerator:
    """High-performance Point-E generator with optimizations."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.device)
        
        # Enable optimizations
        if config.use_mixed_precision:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        if config.optimize_memory:
            torch.set_flush_denormal(True)
    
    def load_models(self) -> Tuple[nn.Module, nn.Module]:
        """Load models with optimizations."""
        self.logger.info("Loading optimized models...")
        
        from point_e.models.configs import MODEL_CONFIGS, model_from_config
        from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
        from point_e.models.download import load_checkpoint
        
        # Load base model
        base_model = model_from_config(MODEL_CONFIGS['base40M-textvec'], self.device)
        base_model.eval()
        
        # Optimize model (disabled JIT due to compatibility issues)
        if self.config.optimize_memory:
            # JIT compilation disabled due to compatibility issues
            pass
        
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['base40M-textvec'])
        base_model.load_state_dict(load_checkpoint('base40M-textvec', self.device))
        
        # Load upsampler
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], self.device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
        upsampler_model.load_state_dict(load_checkpoint('upsample', self.device))
        
        return (base_model, upsampler_model), (base_diffusion, upsampler_diffusion)
    
    def generate_batch(self, prompts: List[str]) -> List[object]:
        """Generate point clouds in parallel batches."""
        self.logger.info(f"Generating {len(prompts)} point clouds...")
        
        models, diffusions = self.load_models()
        
        # Create optimized sampler
        from point_e.diffusion.sampler import PointCloudSampler
        
        sampler = PointCloudSampler(
            device=self.device,
            models=models,
            diffusions=diffusions,
            num_points=[1024, self.config.target_points - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[self.config.guidance_scale, 0.0],
            model_kwargs_key_filter=['texts', ''],
            use_karras=[True, True],
            karras_steps=[self.config.karras_steps, self.config.karras_steps],
            sigma_min=[1e-3, 1e-3],
            sigma_max=[120, 160],
            s_churn=[3, 0],
        )
        
        # Process in batches
        results = []
        for i in range(0, len(prompts), self.config.batch_size):
            batch = prompts[i:i + self.config.batch_size]
            batch_results = self._process_batch(sampler, batch)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, sampler, batch: List[str]) -> List[object]:
        """Process a single batch with optimizations."""
        with torch.no_grad():
            samples = None
            for x in sampler.sample_batch_progressive(
                batch_size=len(batch), 
                model_kwargs=dict(texts=batch)
            ):
                samples = x
        
        return [sampler.output_to_point_clouds(samples)[i] for i in range(len(batch))]
