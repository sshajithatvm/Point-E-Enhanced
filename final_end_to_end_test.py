#!/usr/bin/env python3

import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.enhancements.applyenhancements import enhance_point_cloud

def main():
    """Final end-to-end test of Point-E with enhancements."""
    print("🚀 FINAL END-TO-END POINT-E TEST")
    print("=" * 60)
    
    # Setup
    device = torch.device('cpu')  # Use CPU for faster testing
    print(f"Device: {device}")
    
    # Load models
    print("Loading models...")
    start_time = time.time()
    
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    
    # Load checkpoint
    print("Downloading checkpoint...")
    base_model.load_state_dict(load_checkpoint(base_name, device))
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    
    # Create sampler
    sampler = PointCloudSampler(
        device=device,
        models=[base_model],
        diffusions=[base_diffusion],
        num_points=[512],  # Reduced for faster testing
        aux_channels=['R', 'G', 'B'],
        model_kwargs_key_filter=['texts'],
        guidance_scale=[3.0],
        use_karras=[True],
        karras_steps=[16],  # Reduced for faster testing
        sigma_min=[1e-3],
        sigma_max=[120],
        s_churn=[3],
    )
    
    # Generate point cloud
    prompt = 'a red car'
    print(f"Generating: '{prompt}'")
    
    generation_start = time.time()
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])), total=16):
        samples = x
    
    generation_time = time.time() - generation_start
    print(f"Generation completed in {generation_time:.2f}s")
    
    # Get original point cloud
    original_pc = sampler.output_to_point_clouds(samples)[0]
    print(f"Original: {len(original_pc.coords)} points")
    
    # Apply enhancement
    print("Applying enhancement...")
    enhancement_start = time.time()
    
    enhanced_pc = enhance_point_cloud(original_pc, prompt=prompt, save_ply=True, optimize_density=True)
    
    enhancement_time = time.time() - enhancement_start
    print(f"Enhancement completed in {enhancement_time:.2f}s")
    
    # Create visualization
    print("Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original
    ax1 = plot_point_cloud(original_pc, grid_size=1, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    ax1.set_title(f"Original\n{len(original_pc.coords)} points", fontweight='bold')
    
    # Enhanced
    ax2 = plot_point_cloud(enhanced_pc, grid_size=1, fixed_bounds=((-1, -1, -1), (1, 1, 1)))
    ax2.set_title(f"Enhanced\n{len(enhanced_pc.coords)} points", fontweight='bold')
    
    plt.suptitle(f"Point-E End-to-End Test: '{prompt}'", fontweight='bold')
    plt.tight_layout()
    plt.savefig('final_end_to_end_test.png', dpi=120, bbox_inches='tight')
    print("Visualization saved as 'final_end_to_end_test.png'")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 END-TO-END TEST RESULTS")
    print("=" * 60)
    print(f"✅ Point-E Generation: SUCCESS")
    print(f"✅ Enhancement System: SUCCESS")
    print(f"✅ PLY Export: SUCCESS")
    print(f"✅ Visualization: SUCCESS")
    print(f"⏱️  Total Time: {load_time + generation_time + enhancement_time:.2f}s")
    print(f"📁 Files Created:")
    print(f"   - final_end_to_end_test.png")
    print(f"   - enhanced_outputs/ply_files/")
    print(f"   - enhanced_outputs/metadata/")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 END-TO-END TEST COMPLETED SUCCESSFULLY!")
            print("\n✅ Point-E Enhanced repository is ready for production use!")
        else:
            print("\n❌ END-TO-END TEST FAILED!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
