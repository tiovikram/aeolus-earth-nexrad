#!/usr/bin/env python3
"""
Enhanced runner script for the NEXRAD processing pipeline with AWS S3 data loading.
Supports configurable station selection, time range filtering, and processing options.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from apache_beam.options.pipeline_options import PipelineOptions

# Add src to path
sys.path.insert(0, 'src')

from enhanced_nexrad_pipeline import NexradPipelineOptions, process_nexrad_from_aws, NexradDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_available_stations(date_str: str = None):
    """List available radar stations for a given date."""
    
    if date_str:
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            return
    else:
        # Default to yesterday
        date = datetime.now() - timedelta(days=1)
    
    logger.info(f"Checking available stations for {date.strftime('%Y-%m-%d')}")
    
    data_loader = NexradDataLoader()
    stations = data_loader.get_available_stations(date)
    
    if stations:
        logger.info(f"Found {len(stations)} stations:")
        # Group by region for better readability
        regions = {}
        for station in stations:
            region = station[1]  # Second letter often indicates region
            if region not in regions:
                regions[region] = []
            regions[region].append(station)
        
        for region, region_stations in sorted(regions.items()):
            print(f"\nRegion {region}: {', '.join(sorted(region_stations))}")
    else:
        logger.warning(f"No stations found for {date.strftime('%Y-%m-%d')}")


def get_preset_configurations():
    """Get predefined pipeline configurations for common use cases."""
    
    # Calculate recent dates for examples
    yesterday = datetime.now() - timedelta(days=1)
    week_ago = datetime.now() - timedelta(days=7)
    
    configs = {
        'single_station_recent': {
            'description': 'Single station (KABR) for the last 6 hours',
            'stations': 'KABR',
            'start_time': (yesterday - timedelta(hours=6)).strftime('%Y-%m-%d-%H:%M'),
            'end_time': yesterday.strftime('%Y-%m-%d-%H:%M'),
            'max_files_per_station': 10,
            'grid_size': 100,
            'include_sweeps': '0,1,2'
        },
        'multi_station_recent': {
            'description': 'Multiple stations (KABR,KPDT,KYUX) for the last 3 hours',
            'stations': 'KABR,KPDT,KYUX',
            'start_time': (yesterday - timedelta(hours=3)).strftime('%Y-%m-%d-%H:%M'),
            'end_time': yesterday.strftime('%Y-%m-%d-%H:%M'),
            'max_files_per_station': 5,
            'grid_size': 80,
            'include_sweeps': 'all'
        },
        'high_resolution': {
            'description': 'High resolution processing for detailed analysis',
            'stations': 'KPDT',
            'start_time': (yesterday - timedelta(hours=2)).strftime('%Y-%m-%d-%H:%M'),
            'end_time': yesterday.strftime('%Y-%m-%d-%H:%M'),
            'max_files_per_station': 8,
            'grid_size': 150,
            'include_sweeps': '0,1,2,3'
        },
        'operational': {
            'description': 'Operational processing - multiple stations, recent data',
            'stations': 'KABR,KPDT,KYUX,PGUA',
            'start_time': (yesterday - timedelta(hours=1)).strftime('%Y-%m-%d-%H:%M'),
            'end_time': yesterday.strftime('%Y-%m-%d-%H:%M'),
            'max_files_per_station': 3,
            'grid_size': 100,
            'include_sweeps': '0,1'
        },
        'research': {
            'description': 'Research case study - specific time period',
            'stations': 'KABR,KPDT',
            'start_time': '2024-06-01-18:00',  # Example severe weather event
            'end_time': '2024-06-01-23:59',
            'max_files_per_station': 20,
            'grid_size': 120,
            'include_sweeps': 'all'
        }
    }
    
    return configs


def main():
    """Main function with enhanced configuration options."""
    
    parser = argparse.ArgumentParser(
        description='Enhanced NEXRAD Processing Pipeline with AWS S3 Data Loading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available stations
    python run_enhanced_pipeline.py --list-stations --date 2024-01-01
    
    # Use preset configuration
    python run_enhanced_pipeline.py --preset single_station_recent
    
    # Custom configuration
    python run_enhanced_pipeline.py \\
        --stations KABR,KPDT \\
        --start_time 2024-01-01-12:00 \\
        --end_time 2024-01-01-18:00 \\
        --output_path ./custom_output
    
    # High performance distributed processing
    python run_enhanced_pipeline.py \\
        --preset operational \\
        --runner DataflowRunner \\
        --project my-gcp-project
        """
    )
    
    # Action options
    parser.add_argument('--list-stations', action='store_true',
                       help='List available radar stations for a date')
    parser.add_argument('--date', default=None,
                       help='Date for listing stations (YYYY-MM-DD)')
    parser.add_argument('--list-presets', action='store_true',
                       help='List available preset configurations')
    parser.add_argument('--preset', choices=['single_station_recent', 'multi_station_recent', 
                                           'high_resolution', 'operational', 'research'],
                       help='Use a preset configuration')
    
    # Pipeline configuration (can override presets)
    parser.add_argument('--stations', 
                       help='Comma-separated radar station IDs (e.g., KABR,KPDT)')
    parser.add_argument('--start_time',
                       help='Start time in format YYYY-MM-DD-HH:MM (UTC)')
    parser.add_argument('--end_time',
                       help='End time in format YYYY-MM-DD-HH:MM (UTC)')
    parser.add_argument('--output_path', default='./aws_nexrad_output',
                       help='Output directory path')
    parser.add_argument('--max_files_per_station', type=int, default=10,
                       help='Maximum files per station')
    parser.add_argument('--grid_size', type=int, default=100,
                       help='Cartesian grid size (NxN)')
    parser.add_argument('--include_sweeps', default='0,1,2',
                       help='Comma-separated sweep numbers or "all"')
    
    # Apache Beam options
    parser.add_argument('--runner', default='DirectRunner',
                       help='Apache Beam runner (DirectRunner, DataflowRunner, etc.)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of workers')
    parser.add_argument('--project', help='GCP project (for DataflowRunner)')
    parser.add_argument('--region', default='us-central1',
                       help='GCP region (for DataflowRunner)')
    
    args = parser.parse_args()
    
    # Handle utility actions
    if args.list_stations:
        list_available_stations(args.date)
        return
    
    if args.list_presets:
        configs = get_preset_configurations()
        print("\nAvailable preset configurations:")
        print("=" * 50)
        for name, config in configs.items():
            print(f"\n{name.upper()}:")
            print(f"  Description: {config['description']}")
            print(f"  Stations: {config['stations']}")
            print(f"  Time range: {config['start_time']} to {config['end_time']}")
            print(f"  Grid size: {config['grid_size']}x{config['grid_size']}")
            print(f"  Sweeps: {config['include_sweeps']}")
        return
    
    # Load preset configuration if specified
    pipeline_config = {}
    if args.preset:
        configs = get_preset_configurations()
        if args.preset in configs:
            pipeline_config = configs[args.preset].copy()
            logger.info(f"Using preset configuration: {args.preset}")
            logger.info(f"Description: {pipeline_config.pop('description', 'N/A')}")
        else:
            logger.error(f"Unknown preset: {args.preset}")
            return
    
    # Override with command line arguments
    if args.stations:
        pipeline_config['stations'] = args.stations
    if args.start_time:
        pipeline_config['start_time'] = args.start_time
    if args.end_time:
        pipeline_config['end_time'] = args.end_time
    if args.output_path:
        pipeline_config['output_path'] = args.output_path
    if args.max_files_per_station:
        pipeline_config['max_files_per_station'] = args.max_files_per_station
    if args.grid_size:
        pipeline_config['grid_size'] = args.grid_size
    if args.include_sweeps:
        pipeline_config['include_sweeps'] = args.include_sweeps
    
    # Validate required parameters
    required_params = ['stations', 'start_time', 'end_time']
    missing_params = [p for p in required_params if p not in pipeline_config]
    
    if missing_params:
        logger.error(f"Missing required parameters: {', '.join(missing_params)}")
        logger.info("Use --preset or provide --stations, --start_time, and --end_time")
        return
    
    # Build Apache Beam pipeline arguments
    beam_args = [
        f"--runner={args.runner}",
        f"--direct_num_workers={args.num_workers}",
        "--direct_running_mode=multi_processing",
        "--save_main_session=True"
    ]
    
    # Add DataflowRunner specific options
    if args.runner == 'DataflowRunner':
        if not args.project:
            logger.error("--project is required for DataflowRunner")
            return
        
        beam_args.extend([
            f"--project={args.project}",
            f"--region={args.region}",
            "--temp_location=gs://{}/temp".format(args.project),
            "--staging_location=gs://{}/staging".format(args.project),
            "--job_name=nexrad-processing-{}".format(
                datetime.now().strftime('%Y%m%d-%H%M%S')
            )
        ])
    
    # Add NEXRAD-specific options
    for key, value in pipeline_config.items():
        beam_args.append(f"--{key}={value}")
    
    logger.info("Starting Enhanced NEXRAD Processing Pipeline")
    logger.info(f"Configuration: {pipeline_config}")
    logger.info(f"Beam arguments: {beam_args}")
    
    try:
        # Create pipeline options
        pipeline_options = PipelineOptions(beam_args)
        nexrad_options = pipeline_options.view_as(NexradPipelineOptions)
        
        # Run the pipeline
        process_nexrad_from_aws(nexrad_options)
        
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"Results saved to: {pipeline_config.get('output_path', './aws_nexrad_output')}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()