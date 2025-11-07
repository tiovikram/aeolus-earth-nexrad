#!/usr/bin/env python3
"""
Simple runner for the original NEXRAD pipeline with local Zarr data.
"""

import os
import sys
from apache_beam.options.pipeline_options import PipelineOptions

# Add src to path
sys.path.insert(0, 'src')

def main():
    """Run the original NEXRAD pipeline with local data."""
    
    print("🌦️ Running Original NEXRAD Beam Pipeline")
    print("=" * 50)
    
    # Check that data directory exists
    if not os.path.exists('data'):
        print("❌ Error: 'data' directory not found")
        print("Please ensure you have local NEXRAD Zarr data in the 'data/' directory")
        return 1
    
    # Check what's in the data directory
    data_contents = os.listdir('data')
    print(f"Data directory contents: {data_contents}")
    
    # Import the pipeline after checking data
    from nexrad_beam_pipeline import process_nexrad_data
    
    # Set up pipeline options
    pipeline_options = PipelineOptions([
        '--runner=DirectRunner',
        '--direct_running_mode=multi_processing',
        '--direct_num_workers=2'
    ])
    
    try:
        # Run the pipeline
        print("🚀 Starting pipeline processing...")
        process_nexrad_data(
            input_zarr_path='./data',
            output_path='./out',
            pipeline_options=pipeline_options
        )
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        print("\nNote: The original pipeline expects a specific Zarr data structure.")
        print("For AWS S3 processing, use the enhanced pipeline instead:")
        print("  python run_natural_language_pipeline.py --query 'your request'")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())