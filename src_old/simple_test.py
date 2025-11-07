#!/usr/bin/env python3
"""
Simple test script to verify the core functionality without Apache Beam.
"""

import os
import sys
import numpy as np
import xarray as xr
import json
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelGrid:
    """Simplified model grid definition."""
    
    def __init__(self, nx=100, ny=100, nz=10, 
                 lat_min=39.0, lat_max=41.0, lon_min=-97.0, lon_max=-95.0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
    
    @property
    def x_coords(self) -> np.ndarray:
        return np.linspace(self.lon_min, self.lon_max, self.nx)
    
    @property 
    def y_coords(self) -> np.ndarray:
        return np.linspace(self.lat_min, self.lat_max, self.ny)


# Import improved regridding
from improved_regridding import ImprovedPolarToCartesian


def test_simple_processing():
    """Test the simplified processing pipeline."""
    
    data_path = '../data'
    output_path = '../out'
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Load input data
        logger.info("Loading NEXRAD data...")
        import zarr
        zarr_store = zarr.open_group(data_path, mode='r')
        sites = list(zarr_store.group_keys())
        logger.info(f"Found sites: {sites}")
        
        # Process first site
        site = sites[0]
        logger.info(f"Processing site: {site}")
        
        # Find sweeps with data
        site_group_path = f'{site}'
        ds_site = xr.open_zarr(data_path, group=site_group_path)
        
        # Try to find a sweep with actual radar data
        sweep_paths = []
        for i in range(15):  # Try first 15 sweeps
            try:
                sweep_path = f'{site}/sweep_{i:02d}'
                ds_test = xr.open_zarr(data_path, group=sweep_path)
                if any(var in ds_test.data_vars for var in ['DBZH', 'RHOHV', 'ZDR']):
                    sweep_paths.append(sweep_path)
                    logger.info(f"Found data in {sweep_path}")
                    if len(sweep_paths) >= 2:  # Process 2 sweeps
                        break
            except:
                continue
        
        if not sweep_paths:
            logger.error("No sweeps with radar data found!")
            return False
        
        # Create model grid based on radar location
        ds_sample = xr.open_zarr(data_path, group=sweep_paths[0])
        radar_lat = float(ds_sample.latitude.values)
        radar_lon = float(ds_sample.longitude.values)
        
        model_grid = ModelGrid(
            nx=80, ny=80,
            lat_min=radar_lat - 0.5, lat_max=radar_lat + 0.5,
            lon_min=radar_lon - 0.5, lon_max=radar_lon + 0.5
        )
        
        logger.info(f"Created model grid centered at {radar_lat:.2f}, {radar_lon:.2f}")
        
        # Process sweeps with improved method
        transformer = ImprovedPolarToCartesian(model_grid)
        
        for sweep_path in sweep_paths:
            logger.info(f"Processing {sweep_path}...")
            
            # Load sweep data
            ds = xr.open_zarr(data_path, group=sweep_path)
            
            logger.info(f"  Original dimensions: {dict(ds.sizes)}")
            logger.info(f"  Variables: {list(ds.data_vars.keys())}")
            
            # Transform to cartesian with improved method
            cartesian_ds = transformer.regrid_to_cartesian_improved(ds)
            
            logger.info(f"  Cartesian dimensions: {dict(cartesian_ds.sizes)}")
            logger.info(f"  Cartesian variables: {list(cartesian_ds.data_vars.keys())}")
            
            # Add metadata
            cartesian_ds.attrs.update({
                'site_name': site,
                'sweep_path': sweep_path,
                'processing_method': 'simple_nearest_neighbor'
            })
            
            # Save processed data
            output_file = os.path.join(output_path, f"{site}_{sweep_path.split('/')[-1]}.zarr")
            cartesian_ds.to_zarr(output_file, mode='w')
            
            logger.info(f"  Saved to {output_file}")
            
            # Basic quality metrics
            qc_metrics = {}
            for var in ['DBZH', 'RHOHV', 'ZDR']:
                if var in cartesian_ds.data_vars:
                    data = cartesian_ds[var].values
                    valid_data = data[np.isfinite(data)]
                    if len(valid_data) > 0:
                        qc_metrics[f'{var}_coverage'] = len(valid_data) / data.size
                        qc_metrics[f'{var}_mean'] = float(np.mean(valid_data))
                        qc_metrics[f'{var}_std'] = float(np.std(valid_data))
                        qc_metrics[f'{var}_range'] = [float(np.min(valid_data)), float(np.max(valid_data))]
            
            # Save QC metrics
            qc_file = output_file.replace('.zarr', '_qc.json')
            with open(qc_file, 'w') as f:
                json.dump(qc_metrics, f, indent=2)
            
            logger.info(f"  QC metrics: {qc_metrics}")
        
        logger.info("Processing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_output():
    """Validate the processed output."""
    
    output_path = '../out'
    
    try:
        output_files = [f for f in os.listdir(output_path) if f.endswith('.zarr')]
        logger.info(f"Found {len(output_files)} output files")
        
        for output_file in output_files:
            file_path = os.path.join(output_path, output_file)
            ds = xr.open_zarr(file_path)
            
            logger.info(f"\nValidating {output_file}:")
            logger.info(f"  Dimensions: {dict(ds.sizes)}")
            logger.info(f"  Variables: {list(ds.data_vars.keys())}")
            
            # Check coordinate ranges
            if 'x' in ds.coords and 'y' in ds.coords:
                logger.info(f"  X (lon) range: {ds.x.min().values:.3f} to {ds.x.max().values:.3f}")
                logger.info(f"  Y (lat) range: {ds.y.min().values:.3f} to {ds.y.max().values:.3f}")
            
            # Check data coverage
            for var in ds.data_vars:
                if hasattr(ds[var], 'values'):
                    data = ds[var].values
                    if data.size > 0:
                        valid_fraction = np.sum(np.isfinite(data)) / data.size
                        logger.info(f"  {var} coverage: {valid_fraction:.2%}")
                        if np.any(np.isfinite(data)):
                            valid_data = data[np.isfinite(data)]
                            logger.info(f"  {var} range: {valid_data.min():.2f} to {valid_data.max():.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Output validation failed: {e}")
        return False


if __name__ == '__main__':
    logger.info("Starting simple NEXRAD processing test...")
    
    # Test processing
    if test_simple_processing():
        logger.info("✓ Processing test passed")
        
        # Validate output
        if validate_output():
            logger.info("✓ Output validation passed")
            logger.info("\nAll tests completed successfully!")
        else:
            logger.error("✗ Output validation failed")
    else:
        logger.error("✗ Processing test failed")