#!/usr/bin/env python3
"""
Main runner script for the NEXRAD processing pipeline.
"""

import os
import sys
import argparse
import logging
from apache_beam.options.pipeline_options import PipelineOptions

# Add src to path
sys.path.insert(0, 'src')

from nexrad_beam_pipeline import process_nexrad_data, ModelGrid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Process NEXRAD data using Apache Beam')
    parser.add_argument('--input', default='./data', 
                       help='Input Zarr store path (default: ./data)')
    parser.add_argument('--output', default='./out', 
                       help='Output directory path (default: ./out)')
    parser.add_argument('--runner', default='DirectRunner',
                       help='Apache Beam runner (default: DirectRunner)')
    parser.add_argument('--grid-size', type=int, default=100,
                       help='Model grid size (default: 100)')
    parser.add_argument('--direct-num-workers', type=int, default=2,
                       help='Number of workers for DirectRunner (default: 2)')
    
    args = parser.parse_args()
    
    # Set up pipeline options
    pipeline_options = PipelineOptions([
        f'--runner={args.runner}',
        '--direct_running_mode=multi_processing',
        f'--direct_num_workers={args.direct_num_workers}',
        '--save_main_session=True'
    ])
    
    # Create model grid (will be auto-sized based on data if not specified)
    model_grid = None
    if args.grid_size != 100:
        # Create custom grid size
        from nexrad_beam_pipeline import create_model_grid_from_data
        model_grid = create_model_grid_from_data(args.input)
        model_grid.nx = args.grid_size
        model_grid.ny = args.grid_size
    
    logger.info("Starting NEXRAD processing pipeline...")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Runner: {args.runner}")
    logger.info(f"Grid size: {args.grid_size}x{args.grid_size}")
    
    try:
        # Run the pipeline
        process_nexrad_data(
            input_zarr_path=args.input,
            output_path=args.output,
            pipeline_options=pipeline_options,
            model_grid=model_grid
        )
        
        logger.info("✓ Pipeline completed successfully!")
        
        # Check output
        output_files = [f for f in os.listdir(args.output) if f.endswith('.zarr')]
        qc_files = [f for f in os.listdir(args.output) if f.endswith('_qc.json')]
        
        logger.info(f"Generated {len(output_files)} processed datasets")
        logger.info(f"Generated {len(qc_files)} QC reports")
        
        logger.info("\nTo visualize results, run:")
        logger.info("  cd src && python3 visualize_results.py")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()