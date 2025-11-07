#!/usr/bin/env python3
"""
Test and validation script for the NEXRAD processing pipeline.
"""

import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import json
from typing import Dict, List
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nexrad_beam_pipeline import (
    process_nexrad_data, ModelGrid, QualityController, 
    PolarToCartesianTransform, create_model_grid_from_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineValidator:
    """Validation and testing for the NEXRAD pipeline."""
    
    def __init__(self, data_path: str, output_path: str):
        self.data_path = data_path
        self.output_path = output_path
        
    def test_data_loading(self) -> bool:
        """Test that we can load the input data correctly."""
        try:
            store = xr.open_zarr(self.data_path)
            sites = list(store.group_keys())
            logger.info(f"Successfully loaded data with sites: {sites}")
            
            # Test loading a specific sweep
            test_site = sites[0]
            ds = xr.open_zarr(self.data_path, group=f'{test_site}/sweep_03')
            logger.info(f"Test sweep dimensions: {dict(ds.sizes)}")
            logger.info(f"Test sweep variables: {list(ds.data_vars.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data loading test failed: {e}")
            return False
    
    def test_coordinate_transformation(self) -> bool:
        """Test the polar to cartesian coordinate transformation."""
        try:
            # Load test data
            store = xr.open_zarr(self.data_path)
            test_site = list(store.group_keys())[0]
            ds = xr.open_zarr(self.data_path, group=f'{test_site}/sweep_03')
            
            # Create model grid and transformer
            model_grid = ModelGrid(nx=50, ny=50, nz=5)
            transformer = PolarToCartesianTransform(model_grid)
            
            # Test coordinate transformation
            target_lats, target_lons = transformer.transform_coordinates(ds)
            
            logger.info(f"Original radar location: {ds.latitude.values}, {ds.longitude.values}")
            logger.info(f"Transformed coordinate ranges:")
            logger.info(f"  Latitude: {target_lats.min():.3f} to {target_lats.max():.3f}")
            logger.info(f"  Longitude: {target_lons.min():.3f} to {target_lons.max():.3f}")
            
            # Test regridding
            regridded = transformer.regrid_to_cartesian(ds)
            logger.info(f"Regridded dimensions: {dict(regridded.sizes)}")
            logger.info(f"Regridded variables: {list(regridded.data_vars.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Coordinate transformation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_quality_control(self) -> bool:
        """Test the quality control framework."""
        try:
            # Load test data
            store = xr.open_zarr(self.data_path)
            test_site = list(store.group_keys())[0]
            ds = xr.open_zarr(self.data_path, group=f'{test_site}/sweep_03')
            
            qc = QualityController()
            
            # Test physical bounds validation
            bounds_check = qc.validate_physical_bounds(ds)
            logger.info(f"Physical bounds check: {bounds_check}")
            
            # Test coverage assessment
            coverage = qc.assess_coverage(ds)
            logger.info(f"Coverage metrics: {coverage}")
            
            # Test signal quality metrics
            quality = qc.signal_quality_metrics(ds)
            logger.info(f"Signal quality metrics: {quality}")
            
            return True
            
        except Exception as e:
            logger.error(f"Quality control test failed: {e}")
            return False
    
    def run_pipeline_test(self) -> bool:
        """Run a small test of the full pipeline."""
        try:
            from apache_beam.options.pipeline_options import PipelineOptions
            
            # Create test model grid (smaller for testing)
            model_grid = create_model_grid_from_data(self.data_path)
            model_grid.nx = 50  # Reduce grid size for testing
            model_grid.ny = 50
            
            # Set up pipeline options for local testing
            pipeline_options = PipelineOptions([
                '--runner=DirectRunner',
                '--direct_running_mode=in_memory'
            ])
            
            # Run pipeline
            process_nexrad_data(
                input_zarr_path=self.data_path,
                output_path=self.output_path,
                pipeline_options=pipeline_options,
                model_grid=model_grid
            )
            
            logger.info("Pipeline test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_output(self) -> bool:
        """Validate the output data."""
        try:
            output_files = [f for f in os.listdir(self.output_path) if f.endswith('.zarr')]
            qc_files = [f for f in os.listdir(self.output_path) if f.endswith('_qc.json')]
            
            logger.info(f"Found {len(output_files)} output Zarr files")
            logger.info(f"Found {len(qc_files)} QC metric files")
            
            if not output_files:
                logger.error("No output files found!")
                return False
            
            # Validate a sample output file
            test_file = output_files[0]
            test_path = os.path.join(self.output_path, test_file)
            
            ds = xr.open_zarr(test_path)
            logger.info(f"Sample output file {test_file}:")
            logger.info(f"  Dimensions: {dict(ds.sizes)}")
            logger.info(f"  Variables: {list(ds.data_vars.keys())}")
            logger.info(f"  Coordinates: {list(ds.coords.keys())}")
            
            # Check that coordinates are cartesian
            if 'x' in ds.coords and 'y' in ds.coords:
                logger.info(f"  X range: {ds.x.min().values:.3f} to {ds.x.max().values:.3f}")
                logger.info(f"  Y range: {ds.y.min().values:.3f} to {ds.y.max().values:.3f}")
            
            # Load and display QC metrics
            qc_file = test_file.replace('.zarr', '_qc.json')
            if qc_file in qc_files:
                with open(os.path.join(self.output_path, qc_file), 'r') as f:
                    qc_data = json.load(f)
                logger.info(f"  QC metrics: {qc_data}")
            
            return True
            
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False


def create_visualizations(data_path: str, output_path: str, save_path: str = './visualizations') -> None:
    """Create visualizations to verify the processing."""
    
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Load original data for comparison
        store = xr.open_zarr(data_path)
        test_site = list(store.group_keys())[0]
        original_ds = xr.open_zarr(data_path, group=f'{test_site}/sweep_03')
        
        # Load processed data
        output_files = [f for f in os.listdir(output_path) if f.endswith('.zarr')]
        if not output_files:
            logger.error("No processed data found for visualization")
            return
            
        processed_ds = xr.open_zarr(os.path.join(output_path, output_files[0]))
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Original DBZH in polar coordinates
        if 'DBZH' in original_ds.data_vars:
            im1 = ax1.imshow(original_ds.DBZH.values, aspect='auto', cmap='viridis', vmin=-20, vmax=60)
            ax1.set_title(f'Original DBZH - {test_site} (Polar)')
            ax1.set_xlabel('Range bins')
            ax1.set_ylabel('Azimuth bins')
            plt.colorbar(im1, ax=ax1, label='dBZ')
        
        # Plot 2: Processed DBZH in cartesian coordinates
        if 'DBZH' in processed_ds.data_vars:
            im2 = ax2.imshow(processed_ds.DBZH.values, aspect='auto', cmap='viridis', vmin=-20, vmax=60,
                           extent=[processed_ds.x.min(), processed_ds.x.max(), 
                                 processed_ds.y.min(), processed_ds.y.max()])
            ax2.set_title('Processed DBZH (Cartesian)')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            plt.colorbar(im2, ax=ax2, label='dBZ')
        
        # Plot 3: Original RHOHV
        if 'RHOHV' in original_ds.data_vars:
            im3 = ax3.imshow(original_ds.RHOHV.values, aspect='auto', cmap='plasma', vmin=0, vmax=1)
            ax3.set_title('Original RHOHV (Polar)')
            ax3.set_xlabel('Range bins')
            ax3.set_ylabel('Azimuth bins')
            plt.colorbar(im3, ax=ax3, label='Correlation')
        
        # Plot 4: Processed RHOHV
        if 'RHOHV' in processed_ds.data_vars:
            im4 = ax4.imshow(processed_ds.RHOHV.values, aspect='auto', cmap='plasma', vmin=0, vmax=1,
                           extent=[processed_ds.x.min(), processed_ds.x.max(),
                                 processed_ds.y.min(), processed_ds.y.max()])
            ax4.set_title('Processed RHOHV (Cartesian)')
            ax4.set_xlabel('Longitude')
            ax4.set_ylabel('Latitude')
            plt.colorbar(im4, ax=ax4, label='Correlation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'nexrad_processing_comparison.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Visualizations saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")


def main():
    """Run all tests and create visualizations."""
    
    data_path = './data'
    output_path = './out'
    
    # Create validator
    validator = PipelineValidator(data_path, output_path)
    
    logger.info("Starting NEXRAD pipeline tests...")
    
    # Run tests
    tests = [
        ("Data Loading", validator.test_data_loading),
        ("Coordinate Transformation", validator.test_coordinate_transformation), 
        ("Quality Control", validator.test_quality_control),
        ("Full Pipeline", validator.run_pipeline_test),
        ("Output Validation", validator.validate_output)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASSED" if result else "FAILED"
            logger.info(f"Test {test_name}: {status}")
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    # Create visualizations if pipeline ran successfully
    if any(name == "Full Pipeline" and result for name, result in results):
        logger.info("\nCreating visualizations...")
        create_visualizations(data_path, output_path)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)