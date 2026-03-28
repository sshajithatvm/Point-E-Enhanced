import time
import numpy as np
from optimized_point_e_pipeline import OptimizedPointEGenerator

def run_benchmark():
    # Test multiprocessing performance
    print('Testing multiprocessing performance...')

    # Create pipeline
    pipeline = OptimizedPointEGenerator(
        num_workers=4,
        deterministic=True,
        enable_geometric_quality_enhancements=True
    )

    # Test prompts
    prompts = [
        'a red motorcycle',
        'a sports car',
        'a futuristic robot'
    ]

    print(f'Generating {len(prompts)} point clouds with {pipeline.num_workers} workers...')

    # Time the generation
    start_time = time.time()
    results = pipeline.generate_point_clouds_parallel(prompts, batch_size=1)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / len(prompts)

    print(f'Total time: {total_time:.2f}s')
    print(f'Average time per prompt: {avg_time:.2f}s')
    print(f'Generated {len(results)} point clouds')

    # Verify results
    for i, result in enumerate(results):
        if result['success']:
            pc = result['point_cloud']
            print(f'  Prompt {i+1}: {len(pc.coords)} points generated successfully')
        else:
            print(f'  Prompt {i+1}: Failed - {result.get("error", "Unknown error")}')

if __name__ == '__main__':
    run_benchmark()