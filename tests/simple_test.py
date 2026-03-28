import time
import numpy as np
from optimized_point_e_pipeline import OptimizedPointEGenerator

def run_simple_test():
    print('Testing basic pipeline functionality...')

    # Create pipeline with single worker (no multiprocessing)
    pipeline = OptimizedPointEGenerator(
        num_workers=1,
        deterministic=True,
        enable_geometric_quality_enhancements=True
    )

    # Test single prompt
    prompt = 'a red motorcycle'

    print(f'Generating point cloud for: "{prompt}"')

    # Time the generation
    start_time = time.time()
    result = pipeline.generate_single_point_cloud(prompt)
    end_time = time.time()

    generation_time = end_time - start_time

    if result['success']:
        pc = result['point_cloud']
        print(f'✓ Success! Generated {len(pc.coords)} points in {generation_time:.2f}s')
        print(f'  Point cloud shape: {pc.coords.shape}')
        print(f'  Coordinate range: [{pc.coords.min():.3f}, {pc.coords.max():.3f}]')
    else:
        print(f'✗ Failed: {result.get("error", "Unknown error")}')

    print('\nTesting coordinate preservation in enhancements...')

    # Test that enhancements preserve coordinates
    from point_e.enhancements.geometric_quality_enhancements import enhance_point_cloud_quality

    test_coords = np.random.rand(50, 3) * 2 - 1
    test_pc = type('PointCloud', (), {'coords': test_coords, 'channels': {}})()

    enhanced_pc = enhance_point_cloud_quality(test_pc)

    coords_identical = np.array_equal(test_pc.coords, enhanced_pc.coords)
    count_preserved = len(test_pc.coords) == len(enhanced_pc.coords)

    print(f'  Coordinates identical: {coords_identical}')
    print(f'  Point count preserved: {count_preserved}')

    if coords_identical and count_preserved:
        print('✓ Enhancement functions preserve geometry correctly!')
    else:
        print('✗ Enhancement functions modified geometry!')

if __name__ == '__main__':
    run_simple_test()