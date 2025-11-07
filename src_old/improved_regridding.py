#!/usr/bin/env python3
"""
Improved regridding approach to minimize data loss during polar to cartesian conversion.
"""

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import logging

logger = logging.getLogger(__name__)


class ImprovedPolarToCartesian:
    """Improved coordinate transformation with better data preservation."""
    
    def __init__(self, model_grid):
        self.model_grid = model_grid
    
    def transform_coordinates(self, ds: xr.Dataset):
        """Transform polar to cartesian coordinates."""
        radar_lat = float(ds.latitude.values)
        radar_lon = float(ds.longitude.values)
        
        ranges = ds.range.values
        azimuths = ds.azimuth.values
        
        range_grid, azimuth_grid = np.meshgrid(ranges, azimuths)
        azimuth_rad = np.radians(azimuth_grid)
        
        # More accurate coordinate transformation
        dx = range_grid * np.sin(azimuth_rad)
        dy = range_grid * np.cos(azimuth_rad)
        
        # Earth radius approximation for better accuracy
        R_earth = 6371000  # meters
        lat_displacement = dy / R_earth * 180 / np.pi
        lon_displacement = dx / (R_earth * np.cos(np.radians(radar_lat))) * 180 / np.pi
        
        target_lats = radar_lat + lat_displacement
        target_lons = radar_lon + lon_displacement
        
        return target_lats, target_lons
    
    def regrid_to_cartesian_improved(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Improved regridding with better data preservation.
        
        Key improvements:
        1. Use ALL data points (no decimation)
        2. More sophisticated interpolation
        3. Adaptive distance thresholds
        4. Better handling of missing values
        """
        target_lats, target_lons = self.transform_coordinates(ds)
        
        x_out = self.model_grid.x_coords
        y_out = self.model_grid.y_coords
        
        # Create output grid
        X_out, Y_out = np.meshgrid(x_out, y_out)
        output_points = np.column_stack([Y_out.ravel(), X_out.ravel()])
        
        output_vars = {}
        
        for var_name in ['DBZH', 'RHOHV', 'ZDR', 'PHIDP']:
            if var_name not in ds.data_vars:
                continue
                
            logger.info(f"Processing {var_name}...")
            var_data = ds[var_name].values
            
            # Method 1: Use all valid data points (no decimation)
            valid_mask = (var_data != -33) & np.isfinite(var_data)
            
            if not np.any(valid_mask):
                logger.warning(f"No valid data for {var_name}")
                continue
            
            # Get all valid source points and values
            source_lats = target_lats[valid_mask]
            source_lons = target_lons[valid_mask]  
            source_values = var_data[valid_mask]
            
            logger.info(f"  Using {len(source_values):,} valid points "
                       f"({100*len(source_values)/var_data.size:.1f}% of original)")
            
            # Remove any remaining invalid coordinates
            coord_valid = (np.isfinite(source_lats) & np.isfinite(source_lons) & 
                          np.isfinite(source_values))
            
            source_lats = source_lats[coord_valid]
            source_lons = source_lons[coord_valid]
            source_values = source_values[coord_valid]
            
            if len(source_values) < 10:
                logger.warning(f"Too few valid points for {var_name}: {len(source_values)}")
                continue
            
            # Create source points array
            source_points = np.column_stack([source_lats, source_lons])
            
            try:
                # Method A: Nearest neighbor interpolation (fast, preserves values)
                output_grid_nn = self._nearest_neighbor_interpolation(
                    source_points, source_values, output_points, Y_out.shape
                )
                
                # Method B: Linear interpolation where dense enough
                output_grid_linear = self._adaptive_linear_interpolation(
                    source_points, source_values, output_points, Y_out.shape
                )
                
                # Combine methods: linear where possible, nearest neighbor elsewhere
                output_grid = np.where(np.isfinite(output_grid_linear), 
                                     output_grid_linear, output_grid_nn)
                
                output_vars[var_name] = (('y', 'x'), output_grid)
                
                # Calculate coverage
                coverage = np.sum(np.isfinite(output_grid)) / output_grid.size
                logger.info(f"  Output coverage: {coverage:.1%}")
                
            except Exception as e:
                logger.error(f"Failed to interpolate {var_name}: {e}")
                continue
        
        # Create output dataset
        output_ds = xr.Dataset(
            output_vars,
            coords={
                'x': ('x', x_out),
                'y': ('y', y_out),
                'lon': ('x', x_out),
                'lat': ('y', y_out),
            },
            attrs={
                **ds.attrs,
                'grid_type': 'cartesian',
                'original_radar_lat': float(ds.latitude.values),
                'original_radar_lon': float(ds.longitude.values),
                'regridding_method': 'improved_adaptive'
            }
        )
        
        return output_ds
    
    def _nearest_neighbor_interpolation(self, source_points, source_values, 
                                      output_points, output_shape):
        """Fast nearest neighbor interpolation."""
        
        # Build KD-tree for efficient lookup
        tree = cKDTree(source_points)
        
        # Find nearest neighbors
        distances, indices = tree.query(output_points)
        
        # Interpolate
        interpolated_values = source_values[indices]
        
        # Mask points too far from any data
        # Adaptive threshold based on data density
        median_distance = np.median(distances)
        max_distance = min(0.05, median_distance * 3)  # degrees, ~5km max
        
        mask = distances > max_distance
        interpolated_values[mask] = np.nan
        
        return interpolated_values.reshape(output_shape)
    
    def _adaptive_linear_interpolation(self, source_points, source_values,
                                     output_points, output_shape):
        """Linear interpolation where data density allows."""
        
        try:
            # Use scipy's griddata with linear interpolation
            interpolated = griddata(
                source_points, source_values, 
                output_points[:, [1, 0]],  # griddata expects (x, y) = (lon, lat)
                method='linear', 
                fill_value=np.nan
            )
            
            return interpolated.reshape(output_shape)
            
        except Exception as e:
            logger.warning(f"Linear interpolation failed: {e}")
            return np.full(output_shape, np.nan)
    
    def _optimize_grid_for_radar(self, ds: xr.Dataset, buffer_factor=1.2):
        """
        Create optimized grid that matches radar coverage pattern.
        This reduces wasted grid points in areas with no data.
        """
        target_lats, target_lons = self.transform_coordinates(ds)
        
        # Find actual data extent
        valid_data = ds['DBZH'].values != -33
        valid_lats = target_lats[valid_data]
        valid_lons = target_lons[valid_data]
        
        if len(valid_lats) == 0:
            return self.model_grid
        
        # Create tighter bounds with buffer
        lat_range = valid_lats.max() - valid_lats.min()
        lon_range = valid_lons.max() - valid_lons.min()
        
        lat_buffer = lat_range * (buffer_factor - 1) / 2
        lon_buffer = lon_range * (buffer_factor - 1) / 2
        
        optimized_grid = type(self.model_grid)(
            nx=self.model_grid.nx,
            ny=self.model_grid.ny,
            lat_min=valid_lats.min() - lat_buffer,
            lat_max=valid_lats.max() + lat_buffer,
            lon_min=valid_lons.min() - lon_buffer,
            lon_max=valid_lons.max() + lon_buffer
        )
        
        logger.info(f"Optimized grid bounds: "
                   f"lat [{optimized_grid.lat_min:.3f}, {optimized_grid.lat_max:.3f}], "
                   f"lon [{optimized_grid.lon_min:.3f}, {optimized_grid.lon_max:.3f}]")
        
        return optimized_grid


def test_improved_regridding():
    """Test the improved regridding approach."""
    
    import sys
    import os
    sys.path.append('.')
    
    from simple_test import ModelGrid
    
    logger.info("Testing improved regridding...")
    
    # Load test data
    data_path = '../data'
    ds = xr.open_zarr(data_path, group='KABR/sweep_03')
    
    logger.info(f"Original data: {dict(ds.sizes)}")
    logger.info(f"Variables: {list(ds.data_vars.keys())}")
    
    # Create model grid  
    radar_lat = float(ds.latitude.values)
    radar_lon = float(ds.longitude.values)
    
    model_grid = ModelGrid(
        nx=100, ny=100,  # Higher resolution
        lat_min=radar_lat - 1.0, lat_max=radar_lat + 1.0,
        lon_min=radar_lon - 1.0, lon_max=radar_lon + 1.0
    )
    
    # Test improved transformer
    transformer = ImprovedPolarToCartesian(model_grid)
    
    # Optimize grid for actual data coverage
    optimized_grid = transformer._optimize_grid_for_radar(ds)
    transformer.model_grid = optimized_grid
    
    # Transform data
    result = transformer.regrid_to_cartesian_improved(ds)
    
    logger.info(f"Result dimensions: {dict(result.sizes)}")
    logger.info(f"Result variables: {list(result.data_vars.keys())}")
    
    # Calculate coverage improvements
    for var in ['DBZH', 'RHOHV', 'ZDR']:
        if var in result.data_vars:
            coverage = np.sum(np.isfinite(result[var].values)) / result[var].size
            logger.info(f"{var} final coverage: {coverage:.1%}")
    
    # Save improved result
    output_path = '../out/KABR_sweep_03_improved.zarr'
    result.to_zarr(output_path, mode='w')
    logger.info(f"Saved improved result to {output_path}")
    
    return result


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_improved_regridding()