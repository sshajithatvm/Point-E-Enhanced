import numpy as np
import logging
from typing import Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Point cloud validation result."""
    is_valid: bool
    point_count: int
    has_nan: bool
    has_inf: bool
    bounding_box: Tuple[np.ndarray, np.ndarray]
    spatial_spread: float
    volume: float
    density: float
    errors: list

class PointCloudValidator:
    """Strict point cloud validation for production reliability."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_point_cloud(self, pc, min_points: int = 100, max_points: int = 10000) -> ValidationResult:
        """Validate point cloud with strict criteria."""
        errors = []
        
        # Convert points to numpy array
        try:
            points = np.array(pc.coords)
            if len(points.shape) != 2 or points.shape[1] != 3:
                errors.append(f"Invalid point cloud shape: {points.shape}")
                return self._create_invalid_result(pc, errors)
        except Exception as e:
            errors.append(f"Failed to convert points to array: {e}")
            return self._create_invalid_result(pc, errors)
        
        # Check point count
        point_count = len(points)
        if point_count < min_points:
            errors.append(f"Too few points: {point_count} < {min_points}")
        elif point_count > max_points:
            errors.append(f"Too many points: {point_count} > {max_points}")
        
        # Check for NaN values
        has_nan = np.isnan(points).any()
        if has_nan:
            nan_count = np.isnan(points).sum()
            errors.append(f"Contains {nan_count} NaN values")
        
        # Check for infinite values
        has_inf = np.isinf(points).any()
        if has_inf:
            inf_count = np.isinf(points).sum()
            errors.append(f"Contains {inf_count} infinite values")
        
        # Check bounding box
        try:
            min_coords = np.nanmin(points, axis=0)
            max_coords = np.nanmax(points, axis=0)
            
            # Check for degenerate bounding box
            bounding_box_size = max_coords - min_coords
            if np.any(bounding_box_size <= 0):
                errors.append(f"Degenerate bounding box: {bounding_box_size}")
            
            # Check spatial spread
            spatial_spread = np.linalg.norm(bounding_box_size)
            if spatial_spread < 0.01:  # Very small spread
                errors.append(f"Insufficient spatial spread: {spatial_spread:.6f}")
            
            # Calculate volume
            volume = np.prod(bounding_box_size)
            if volume <= 0:
                errors.append(f"Invalid volume: {volume}")
            
            # Calculate density with safety checks
            min_volume = 1e-8  # Minimum volume threshold
            safe_volume = max(volume, min_volume)
            density = point_count / safe_volume
            if density < 1.0:  # Very low density
                errors.append(f"Very low density: {density:.3f}")
            
        except Exception as e:
            errors.append(f"Failed to calculate bounding box: {e}")
        
        # Check coordinate ranges
        try:
            coord_ranges = {
                'x': (points[:, 0].min(), points[:, 0].max()),
                'y': (points[:, 1].min(), points[:, 1].max()),
                'z': (points[:, 2].min(), points[:, 2].max())
            }
            
            for axis, (min_val, max_val) in coord_ranges.items():
                if abs(min_val) > 10 or abs(max_val) > 10:
                    errors.append(f"Extreme {axis} range: [{min_val:.3f}, {max_val:.3f}]")
                if min_val == max_val:
                    errors.append(f"Zero {axis} range: [{min_val:.3f}, {max_val:.3f}]")
        except Exception as e:
            errors.append(f"Failed to check coordinate ranges: {e}")
        
        # Check for duplicate points
        try:
            unique_points = np.unique(points, axis=0)
            duplicate_count = len(points) - len(unique_points)
            if duplicate_count > len(points) * 0.1:  # More than 10% duplicates
                errors.append(f"Too many duplicates: {duplicate_count} ({duplicate_count/len(points)*100:.1f}%)")
        except Exception as e:
            errors.append(f"Failed to check duplicates: {e}")
        
        # Final validation result
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            point_count=point_count,
            has_nan=has_nan,
            has_inf=has_inf,
            bounding_box=(min_coords if not errors else np.array([0, 0, 0]), 
                        max_coords if not errors else np.array([0, 0, 0])),
            spatial_spread=spatial_spread if not errors else 0.0,
            volume=volume if not errors else 0.0,
            density=density if not errors else 0.0,
            errors=errors
        )
    
    def _create_invalid_result(self, pc, errors: list) -> ValidationResult:
        """Create invalid result for failed validation."""
        return ValidationResult(
            is_valid=False,
            point_count=0,
            has_nan=True,
            has_inf=True,
            bounding_box=(np.array([0, 0, 0]), np.array([0, 0, 0])),
            spatial_spread=0.0,
            volume=0.0,
            density=0.0,
            errors=errors
        )
    
    def print_validation_result(self, result: ValidationResult, pc_name: str = "Point Cloud"):
        """Print detailed validation result."""
        print(f"\n🔍 VALIDATION RESULT: {pc_name}")
        print("=" * 50)
        
        # Overall result
        status = "✅ PASS" if result.is_valid else "❌ FAIL"
        print(f"Overall Status: {status}")
        
        # Basic stats
        print(f"\n📊 Basic Statistics:")
        print(f"  Point Count: {result.point_count}")
        print(f"  Has NaN: {result.has_nan}")
        print(f"  Has Inf: {result.has_inf}")
        
        # Bounding box
        if result.is_valid:
            print(f"\n📦 Bounding Box:")
            print(f"  Min: [{result.bounding_box[0][0]:.3f}, {result.bounding_box[0][1]:.3f}, {result.bounding_box[0][2]:.3f}]")
            print(f"  Max: [{result.bounding_box[1][0]:.3f}, {result.bounding_box[1][1]:.3f}, {result.bounding_box[1][2]:.3f}]")
            print(f"  Size: [{result.bounding_box[1][0] - result.bounding_box[0][0]:.3f}, "
                  f"{result.bounding_box[1][1] - result.bounding_box[0][1]:.3f}, "
                  f"{result.bounding_box[1][2] - result.bounding_box[0][2]:.3f}]")
        
        # Spatial metrics
        if result.is_valid:
            print(f"\n📏 Spatial Metrics:")
            print(f"  Spread: {result.spatial_spread:.6f}")
            print(f"  Volume: {result.volume:.6f}")
            print(f"  Density: {result.density:.3f} points/unit³")
        
        # Errors
        if result.errors:
            print(f"\n❌ Validation Errors:")
            for i, error in enumerate(result.errors, 1):
                print(f"  {i}. {error}")
        
        print("=" * 50)
        
        return result.is_valid
