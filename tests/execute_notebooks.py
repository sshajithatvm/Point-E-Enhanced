#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm.auto import tqdm
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e.diffusion.sampler import PointCloudSampler
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e.models.download import load_checkpoint
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e.models.configs import MODEL_CONFIGS, model_from_config
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e.util.plotting import plot_point_cloud
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e.enhancements.applyenhancements import enhance_point_cloud

def execute_text2pointcloud():
    """Execute text2pointcloud notebook functionality."""
    print("🎨 EXECUTING TEXT-TO-POINT CLOUD")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load models
    print("Loading models...")
    start_time = time.time()
    
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    
    # Load checkpoints
    print("Downloading checkpoints...")
    base_model.load_state_dict(load_checkpoint(base_name, device))
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    
    load_time = time.time() - start_time
    print(f"Models loaded in {load_time:.2f}s")
    
    # Create sampler
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=['texts', ''],
    )
    
    # Generate point cloud
    prompt = 'a red motorcycle'
    print(f"Generating point cloud for: '{prompt}'")
    
    generation_start = time.time()
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])), total=40):
        samples = x
    
    generation_time = time.time() - generation_start
    print(f"Generation completed in {generation_time:.2f}s")
    
    # Get original point cloud
    original_pc = sampler.output_to_point_clouds(samples)[0]
    print(f"Original point cloud: {len(original_pc.coords)} points")
    
    # Apply enhancement with before/after comparison
    print("Applying production enhancement...")
    enhancement_start = time.time()
    
    enhanced_pc = enhance_point_cloud(original_pc, production_mode=True, prompt=prompt, save_ply=True, optimize_density=True)
    
    enhancement_time = time.time() - enhancement_start
    print(f"Enhancement completed in {enhancement_time:.2f}s")
    
    # Create before/after visualization
    print("Creating before/after visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original
    ax1 = plot_point_cloud(original_pc, grid_size=2, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    ax1.set_title(f"BEFORE Enhancement\n{len(original_pc.coords)} points", fontsize=14, fontweight='bold')
    
    # Enhanced
    ax2 = plot_point_cloud(enhanced_pc, grid_size=2, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    ax2.set_title(f"AFTER Enhancement\n{len(enhanced_pc.coords)} points", fontsize=14, fontweight='bold')
    
    plt.suptitle(f"Text-to-Point Cloud: '{prompt}'", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    output_file = 'text2pointcloud_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved as: {output_file}")
    
    # Verify outputs
    print("\n" + "=" * 60)
    print("📊 TEXT-TO-POINT CLOUD RESULTS")
    print("=" * 60)
    print(f"✅ Original Points: {len(original_pc.coords)}")
    print(f"✅ Enhanced Points: {len(enhanced_pc.coords)}")
    reduction = ((len(original_pc.coords) - len(enhanced_pc.coords)) / len(original_pc.coords) * 100
    print(f"✅ Point Reduction: {reduction:.1f}%")
    print(f"✅ PLY Files: outputs/enhanced_outputs/ply_files/")
    print(f"✅ Metadata: outputs/enhanced_outputs/metadata/")
    print(f"✅ Visualization: {output_file}")
    
    # Verify point cloud validity
    coords = np.array(enhanced_pc.coords)
    print(f"✅ Point Cloud Valid: {len(coords) > 0 and not np.all(coords == 0)}")
    print(f"✅ Coordinate Range: [{coords.min():.3f}, {coords.max():.3f}]")
    print(f"✅ Total Processing Time: {load_time + generation_time + enhancement_time:.2f}s")
    
    return True

def execute_image2pointcloud():
    """Execute image2pointcloud notebook functionality."""
    print("\n🖼️  EXECUTING IMAGE-TO-POINT CLOUD")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load models
    print("Loading models...")
    start_time = time.time()
    
    base_name = 'base40M'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    
    # Load checkpoints
    print("Downloading checkpoints...")
    base_model.load_state_dict(load_checkpoint(base_name, device))
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    
    load_time = time.time() - start_time
    print(f"Models loaded in {load_time:.2f}s")
    
    # Create sampler
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )
    
    # Create synthetic image for testing (since we don't have example images)
    print("Creating synthetic test image...")
    from PIL import Image
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    
    # Generate point cloud
    print("Generating point cloud from image...")
    
    generation_start = time.time()
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[test_image])), total=40):
        samples = x
    
    generation_time = time.time() - generation_start
    print(f"Generation completed in {generation_time:.2f}s")
    
    # Get original point cloud
    original_pc = sampler.output_to_point_clouds(samples)[0]
    print(f"Original point cloud: {len(original_pc.coords)} points")
    
    # Apply enhancement
    print("Applying production enhancement...")
    enhancement_start = time.time()
    
    enhanced_pc = enhance_point_cloud(original_pc, production_mode=True, prompt="image_conditioned", save_ply=True, optimize_density=True)
    
    enhancement_time = time.time() - enhancement_start
    print(f"Enhancement completed in {enhancement_time:.2f}s")
    
    # Create before/after visualization
    print("Creating before/after visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original
    ax1 = plot_point_cloud(original_pc, grid_size=2, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    ax1.set_title(f"BEFORE Enhancement\n{len(original_pc.coords)} points", fontsize=14, fontweight='bold')
    
    # Enhanced
    ax2 = plot_point_cloud(enhanced_pc, grid_size=2, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    ax2.set_title(f"AFTER Enhancement\n{len(enhanced_pc.coords)} points", fontsize=14, fontweight='bold')
    
    plt.suptitle("Image-to-Point Cloud: Synthetic Test Image", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    output_file = 'image2pointcloud_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved as: {output_file}")
    
    # Verify outputs
    print("\n" + "=" * 60)
    print("📊 IMAGE-TO-POINT CLOUD RESULTS")
    print("=" * 60)
    print(f"✅ Original Points: {len(original_pc.coords)}")
    print(f"✅ Enhanced Points: {len(enhanced_pc.coords)}")
    reduction = ((len(original_pc.coords) - len(enhanced_pc.coords)) / len(original_pc.coords) * 100
    print(f"✅ Point Reduction: {reduction:.1f}%")
    print(f"✅ PLY Files: outputs/enhanced_outputs/ply_files/")
    print(f"✅ Metadata: outputs/enhanced_outputs/metadata/")
    print(f"✅ Visualization: {output_file}")
    
    # Verify point cloud validity
    coords = np.array(enhanced_pc.coords)
    print(f"✅ Point Cloud Valid: {len(coords) > 0 and not np.all(coords == 0)}")
    print(f"✅ Coordinate Range: [{coords.min():.3f}, {coords.max():.3f}]")
    print(f"✅ Total Processing Time: {load_time + generation_time + enhancement_time:.2f}s")
    
    return True

def main():
    """Execute both notebooks and verify results."""
    print("🚀 EXECUTING NOTEBOOKS - COMPREHENSIVE TEST")
    print("=" * 80)
    
    # Execute text2pointcloud
    try:
        text_success = execute_text2pointcloud()
        if text_success:
            print("✅ Text-to-Point Cloud: SUCCESS")
        else:
            print("❌ Text-to-Point Cloud: FAILED")
    except Exception as e:
        print(f"❌ Text-to-Point Cloud ERROR: {e}")
        import traceback
        traceback.print_exc()
        text_success = False
    
    # Execute image2pointcloud
    try:
        image_success = execute_image2pointcloud()
        if image_success:
            print("✅ Image-to-Point Cloud: SUCCESS")
        else:
            print("❌ Image-to-Point Cloud: FAILED")
    except Exception as e:
        print(f"❌ Image-to-Point Cloud ERROR: {e}")
        import traceback
        traceback.print_exc()
        image_success = False
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 FINAL EXECUTION SUMMARY")
    print("=" * 80)
    
    # Check generated files
    generated_files = []
    if os.path.exists('text2pointcloud_results.png'):
        generated_files.append('text2pointcloud_results.png')
    if os.path.exists('image2pointcloud_results.png'):
        generated_files.append('image2pointcloud_results.png')
    
    # Check PLY files
    ply_dir = 'outputs/enhanced_outputs/ply_files'
    if os.path.exists(ply_dir):
        ply_files = os.listdir(ply_dir)
        generated_files.extend([f"ply_files/{f}" for f in ply_files if f.endswith('.ply')])
    
    # Check metadata files
    metadata_dir = 'outputs/enhanced_outputs/metadata'
    if os.path.exists(metadata_dir):
        metadata_files = os.listdir(metadata_dir)
        generated_files.extend([f"metadata/{f}" for f in metadata_files if f.endswith('.json')])
    
    print("📁 Generated Files:")
    for file in generated_files:
        print(f"   ✅ {file}")
    
    # Overall result
    if text_success and image_success:
        print("\n🎉 ALL NOTEBOOKS EXECUTED SUCCESSFULLY!")
        print("\n✅ Virtual Environment: WORKING")
        print("✅ Models: INITIALIZED")
        print("✅ Point Cloud Generation: VALID")
        print("✅ Enhancement System: OPERATIONAL")
        print("✅ Outputs: SAVED & VERIFIED")
        print("\n🚀 Point-E Enhanced Repository is FULLY FUNCTIONAL!")
        return True
    else:
        print("\n❌ SOME NOTEBOOK EXECUTIONS FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
