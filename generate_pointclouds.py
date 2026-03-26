"""
Optimized Point-E Inference Script
Generates point cloud outputs with CPU-friendly parameters
"""

import torch
import numpy as np
from pathlib import Path

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
from point_e.enhancements.applyenhancements import enhance_point_cloud

print("=" * 60)
print("Point-E Text-to-3D Generation Demo")
print("=" * 60)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

# Create output directory
output_dir = Path("point_cloud_outputs")
output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir}")
print()

# Load models
print("Loading models...")
print("  - Creating base model (base40M-textvec)...")
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print("  - Creating upsampler model...")
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print("  - Loading base checkpoint...")
base_model.load_state_dict(load_checkpoint(base_name, device))

print("  - Loading upsampler checkpoint...")
upsampler_model.load_state_dict(load_checkpoint('upsample', device))
print("✓ Models loaded successfully")
print()

# Setup sampler with reduced steps for CPU performance
print("Setting up sampler (with reduced diffusion steps for CPU)...")
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''),
    karras_steps=(32, 32),  # Reduced from (64, 64) for faster CPU inference
)
print("✓ Sampler initialized (32 steps per stage)")
print()

# Test 1: Text-to-3D Generation
print("-" * 60)
print("TEST 1: Text-to-3D Generation")
print("-" * 60)

test_prompts = [
    'a red motorcycle',
    'a blue cube',
    'a purple sphere'
]

for idx, prompt in enumerate(test_prompts, 1):
    print(f"\n{idx}. Generating: '{prompt}'")
    try:
        # Generate point cloud
        samples = None
        for x in sampler.sample_batch_progressive(
            batch_size=1, 
            model_kwargs=dict(texts=[prompt])
        ):
            samples = x
        
        if samples is None:
            print("   ✗ FAILED: No samples generated")
            continue
        
        # Convert to point cloud
        pc = sampler.output_to_point_clouds(samples)[0]
        print(f"   ✓ Point cloud generated: {len(pc.coords)} points")
        
        # Enhance point cloud
        enhanced_pc = enhance_point_cloud(pc)
        print(f"   ✓ Point cloud enhanced")
        
        # Save point cloud
        output_file = output_dir / f"text_{idx}_'{prompt.replace(' ', '_')}.ply"
        enhanced_pc.save(str(output_file))
        print(f"   ✓ Saved to: {output_file}")
        
        # Verify output
        if len(pc.coords) > 0:
            print(f"   ✓ VALID OUTPUT:")
            print(f"     - Points: {len(pc.coords)}")
            print(f"     - Coords range: X[{pc.coords[:, 0].min():.2f}, {pc.coords[:, 0].max():.2f}]")
            print(f"     - Channels: {list(pc.channels.keys())}")
        else:
            print("   ✗ INVALID: Empty point cloud")
    
    except KeyboardInterrupt:
        print("   ✗ INTERRUPTED")
        raise
    except Exception as e:
        print(f"   ✗ ERROR: {e}")

print("\n" + "=" * 60)
print("Point-E Demo Complete!")
print(f"Output files saved to: {output_dir.resolve()}")
print("=" * 60)
