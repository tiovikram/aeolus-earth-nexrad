#!/usr/bin/env python3
"""
Enhanced Apache Beam pipeline for processing NEXRAD data with configurable
AWS S3 data loading, station selection, and time range filtering.
"""

import os
import re
import tempfile
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Iterable
from dataclasses import dataclass
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import xarray as xr
import xradar
import numpy as np
import pandas as pd

# Import our existing components with proper path handling
import sys
import os

# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the current directory to sys.path for imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from nexrad_beam_pipeline import (
    ModelGrid, QualityController, PolarToCartesianTransform,
    create_model_grid_from_data
)
from improved_regridding import ImprovedPolarToCartesian

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NexradPipelineOptions(PipelineOptions):
    """Custom pipeline options for NEXRAD processing."""
    
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            '--stations',
            dest='stations',
            default='KABR,KPDT',
            help='Comma-separated list of radar station IDs (e.g., KABR,KPDT,KYUX)'
        )
        parser.add_argument(
            '--start_time',
            dest='start_time',
            required=True,
            help='Start time in format YYYY-MM-DD-HH:MM (UTC)'
        )
        parser.add_argument(
            '--end_time', 
            dest='end_time',
            required=True,
            help='End time in format YYYY-MM-DD-HH:MM (UTC)'
        )
        parser.add_argument(
            '--output_path',
            dest='output_path',
            default='./nexrad_output',
            help='Output path for processed data'
        )
        parser.add_argument(
            '--max_files_per_station',
            dest='max_files_per_station',
            type=int,
            default=24,
            help='Maximum files per station to process (default: 24)'
        )
        parser.add_argument(
            '--grid_size',
            dest='grid_size',
            type=int,
            default=100,
            help='Cartesian grid size (default: 100x100)'
        )
        parser.add_argument(
            '--include_sweeps',
            dest='include_sweeps',
            default='all',
            help='Comma-separated sweep numbers to include (e.g., 0,1,2) or "all"'
        )


@dataclass
class NexradFileInfo:
    """Information about a NEXRAD file on AWS S3."""
    
    station_id: str
    timestamp: datetime
    s3_key: str
    size: int
    
    @property
    def filename(self) -> str:
        return os.path.basename(self.s3_key)


class NexradDataLoader:
    """Loads NEXRAD Level II data from AWS S3 based on station and time criteria."""
    
    def __init__(self, bucket_name: str = 'unidata-nexrad-level2'):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    def get_available_stations(self, date: datetime) -> List[str]:
        """Get list of available radar stations for a given date."""
        
        date_prefix = date.strftime('%Y/%m/%d')
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f'{date_prefix}/',
                Delimiter='/'
            )
            
            if 'CommonPrefixes' not in response:
                return []
            
            stations = [
                prefix['Prefix'].split('/')[-2]
                for prefix in response['CommonPrefixes']
                if len(prefix['Prefix'].split('/')[-2]) == 4  # Valid station IDs are 4 chars
            ]
            
            return sorted(stations)
            
        except Exception as e:
            logger.error(f"Error listing stations for {date_prefix}: {e}")
            return []
    
    def find_files_for_station_and_timerange(
        self, 
        station_id: str, 
        start_time: datetime, 
        end_time: datetime,
        max_files: Optional[int] = None
    ) -> List[NexradFileInfo]:
        """Find NEXRAD files for a station within the specified time range."""
        
        files = []
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            daily_files = self._get_files_for_station_and_date(station_id, current_date)
            
            # Filter by time range
            for file_info in daily_files:
                if start_time <= file_info.timestamp <= end_time:
                    files.append(file_info)
                    
                    if max_files and len(files) >= max_files:
                        return files[:max_files]
            
            current_date += timedelta(days=1)
        
        logger.info(f"Found {len(files)} files for station {station_id} "
                   f"between {start_time} and {end_time}")
        
        return sorted(files, key=lambda x: x.timestamp)
    
    def _get_files_for_station_and_date(self, station_id: str, date) -> List[NexradFileInfo]:
        """Get all files for a station on a specific date."""
        
        date_str = date.strftime('%Y/%m/%d')
        prefix = f'{date_str}/{station_id}/'
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            files = []
            for obj in response['Contents']:
                # Parse filename to get timestamp
                # Format: STATION_YYYYMMDD_HHMMSS_V06
                filename = os.path.basename(obj['Key'])
                timestamp = self._parse_nexrad_filename(filename)
                
                if timestamp:
                    files.append(NexradFileInfo(
                        station_id=station_id,
                        timestamp=timestamp,
                        s3_key=obj['Key'],
                        size=obj['Size']
                    ))
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files for {station_id} on {date}: {e}")
            return []
    
    def _parse_nexrad_filename(self, filename: str) -> Optional[datetime]:
        """Parse NEXRAD filename to extract timestamp."""
        
        # Pattern: STATION_YYYYMMDD_HHMMSS_V06
        pattern = r'([A-Z]{4})(\d{8})_(\d{6})_V\d{2}'
        match = re.match(pattern, filename)
        
        if not match:
            return None
            
        try:
            date_str = match.group(2)  # YYYYMMDD
            time_str = match.group(3)  # HHMMSS
            
            timestamp_str = f"{date_str}{time_str}"
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
            
            return timestamp
            
        except ValueError:
            return None
    
    def download_file(self, file_info: NexradFileInfo, local_path: str) -> bool:
        """Download a NEXRAD file from S3 to local path."""
        
        try:
            self.s3_client.download_file(
                self.bucket_name,
                file_info.s3_key,
                local_path
            )
            
            # Verify download
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to download {file_info.s3_key}: {e}")
            return False


class ProcessNexradFile(beam.DoFn):
    """Beam DoFn to process individual NEXRAD files from AWS S3."""
    
    def __init__(self, model_grid: ModelGrid, include_sweeps: List[str], qc_enabled: bool = True):
        self.model_grid = model_grid
        self.include_sweeps = include_sweeps
        self.qc_enabled = qc_enabled
        self.transformer = None
        self.qc = None
        self.data_loader = None
    
    def setup(self):
        """Initialize components."""
        self.transformer = ImprovedPolarToCartesian(self.model_grid)
        self.qc = QualityController()
        self.data_loader = NexradDataLoader()
    
    def process(self, file_info: NexradFileInfo) -> Iterable[Tuple[str, xr.Dataset, Dict]]:
        """Process a single NEXRAD file from S3."""
        
        logger.info(f"Processing {file_info.station_id} file: {file_info.filename}")
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(suffix='.nexrad', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Download file from S3
            if not self.data_loader.download_file(file_info, temp_path):
                logger.error(f"Failed to download {file_info.filename}")
                return
            
            # Load with xradar
            dt = xradar.io.open_nexradlevel2_datatree(temp_path)
            
            # Process each sweep
            available_sweeps = list(dt.children.keys())
            logger.info(f"Available sweeps: {available_sweeps}")
            
            # Filter sweeps based on include_sweeps parameter
            if self.include_sweeps == ['all']:
                sweeps_to_process = available_sweeps[:3]  # Process first 3 for efficiency
            else:
                sweeps_to_process = [
                    sweep for sweep in available_sweeps 
                    if any(f'sweep_{num:02d}' in sweep for num in map(int, self.include_sweeps))
                ][:3]  # Limit to 3 sweeps
            
            for sweep_name in sweeps_to_process:
                try:
                    sweep_node = dt[sweep_name]
                    if sweep_node.ds is None:
                        continue
                    
                    sweep_ds = sweep_node.ds
                    
                    # Check if sweep has radar data
                    radar_vars = ['DBZH', 'RHOHV', 'ZDR', 'PHIDP', 'VRADH']
                    if not any(var in sweep_ds.data_vars for var in radar_vars):
                        logger.info(f"No radar variables in {sweep_name}, skipping")
                        continue
                    
                    # Add station metadata from root
                    if dt.ds is not None:
                        sweep_ds = sweep_ds.assign_coords({
                            'latitude': dt.ds.latitude,
                            'longitude': dt.ds.longitude,
                            'altitude': dt.ds.altitude
                        })
                        sweep_ds.attrs.update(dt.ds.attrs)
                    
                    # Quality control
                    qc_metrics = {'file_info': {
                        'station_id': file_info.station_id,
                        'timestamp': file_info.timestamp.isoformat(),
                        'filename': file_info.filename,
                        'size_bytes': file_info.size
                    }}
                    
                    if self.qc_enabled:
                        qc_metrics.update(self.qc.validate_physical_bounds(sweep_ds))
                        qc_metrics.update(self.qc.assess_coverage(sweep_ds))
                        qc_metrics.update(self.qc.signal_quality_metrics(sweep_ds))
                    
                    # Transform to cartesian coordinates
                    try:
                        cartesian_ds = self.transformer.regrid_to_cartesian_improved(sweep_ds)
                        
                        # Add processing metadata
                        cartesian_ds.attrs.update({
                            'station_id': file_info.station_id,
                            'sweep_name': sweep_name,
                            'original_timestamp': file_info.timestamp.isoformat(),
                            'processing_timestamp': datetime.now().isoformat(),
                            'source_file': file_info.filename,
                            's3_key': file_info.s3_key
                        })
                        
                        # Create unique identifier
                        output_id = f"{file_info.station_id}_{file_info.timestamp.strftime('%Y%m%d_%H%M%S')}_{sweep_name}"
                        
                        yield (output_id, cartesian_ds, qc_metrics)
                        
                    except Exception as e:
                        logger.error(f"Failed to transform {sweep_name}: {e}")
                        continue
                
                except Exception as e:
                    logger.error(f"Error processing sweep {sweep_name}: {e}")
                    continue
            
            # Close datatree
            dt.close()
            
        except Exception as e:
            logger.error(f"Error processing file {file_info.filename}: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def create_file_list(pipeline_options: NexradPipelineOptions) -> List[NexradFileInfo]:
    """Create list of NEXRAD files to process based on pipeline options."""
    
    # Parse options
    stations = [s.strip().upper() for s in pipeline_options.stations.split(',')]
    
    try:
        start_time = datetime.strptime(pipeline_options.start_time, '%Y-%m-%d-%H:%M')
        end_time = datetime.strptime(pipeline_options.end_time, '%Y-%m-%d-%H:%M')
    except ValueError as e:
        raise ValueError(f"Invalid time format. Use YYYY-MM-DD-HH:MM. Error: {e}")
    
    if start_time >= end_time:
        raise ValueError("Start time must be before end time")
    
    logger.info(f"Searching for files from stations {stations}")
    logger.info(f"Time range: {start_time} to {end_time}")
    
    # Initialize data loader
    data_loader = NexradDataLoader()
    
    all_files = []
    
    for station in stations:
        try:
            station_files = data_loader.find_files_for_station_and_timerange(
                station, start_time, end_time, 
                max_files=pipeline_options.max_files_per_station
            )
            
            all_files.extend(station_files)
            logger.info(f"Found {len(station_files)} files for {station}")
            
        except Exception as e:
            logger.error(f"Error finding files for station {station}: {e}")
            continue
    
    logger.info(f"Total files to process: {len(all_files)}")
    
    if not all_files:
        raise ValueError("No NEXRAD files found matching the specified criteria")
    
    return all_files


def process_nexrad_from_aws(pipeline_options: NexradPipelineOptions) -> None:
    """
    Main function to process NEXRAD data from AWS S3 with configurable parameters.
    """
    
    logger.info("Starting Enhanced NEXRAD Processing Pipeline")
    logger.info(f"Stations: {pipeline_options.stations}")
    logger.info(f"Time range: {pipeline_options.start_time} to {pipeline_options.end_time}")
    logger.info(f"Output path: {pipeline_options.output_path}")
    
    # Validate and parse include_sweeps
    if pipeline_options.include_sweeps.lower() == 'all':
        include_sweeps = ['all']
    else:
        include_sweeps = [s.strip() for s in pipeline_options.include_sweeps.split(',')]
    
    # Create output directory (only for local paths)
    if not pipeline_options.output_path.startswith('s3://'):
        os.makedirs(pipeline_options.output_path, exist_ok=True)
    
    # Create file list
    try:
        file_list = create_file_list(pipeline_options)
    except Exception as e:
        logger.error(f"Failed to create file list: {e}")
        return
    
    # Create model grid (for now, use a default that will work for all stations)
    # In production, you might want to create station-specific grids
    model_grid = ModelGrid(
        nx=pipeline_options.grid_size,
        ny=pipeline_options.grid_size,
        nz=10,
        lat_min=20.0,  # Cover CONUS
        lat_max=50.0,
        lon_min=-130.0,
        lon_max=-60.0
    )
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        
        logger.info(f"Processing {len(file_list)} files with Apache Beam")
        
        # Process files
        processed_data = (
            pipeline
            | 'Create file list' >> beam.Create(file_list)
            | 'Process NEXRAD files' >> beam.ParDo(
                ProcessNexradFile(model_grid, include_sweeps)
            )
        )
        
        # Save processed datasets
        def save_dataset(element):
            """Save processed dataset and QC metrics to local or S3 storage."""
            output_id, dataset, qc_metrics = element
            
            try:
                is_s3_output = pipeline_options.output_path.startswith('s3://')
                
                if is_s3_output:
                    # S3 output path
                    import s3fs
                    
                    # Parse S3 path
                    s3_path = pipeline_options.output_path.replace('s3://', '')
                    bucket_name, prefix = s3_path.split('/', 1) if '/' in s3_path else (s3_path, '')
                    
                    # Create S3 filesystem
                    s3_fs = s3fs.S3FileSystem()
                    
                    # Save dataset to S3
                    zarr_s3_path = f"s3://{bucket_name}/{prefix}/{output_id}.zarr" if prefix else f"s3://{bucket_name}/{output_id}.zarr"
                    dataset.to_zarr(zarr_s3_path, storage_options={'s3': s3_fs}, mode='w')
                    
                    # Save QC metrics to S3
                    qc_s3_path = f"s3://{bucket_name}/{prefix}/{output_id}_qc.json" if prefix else f"s3://{bucket_name}/{output_id}_qc.json"
                    
                    import json
                    qc_json = json.dumps(qc_metrics, indent=2, default=str)
                    
                    with s3_fs.open(qc_s3_path, 'w') as f:
                        f.write(qc_json)
                    
                    logger.info(f"Saved {output_id} -> {zarr_s3_path}")
                    
                    return {
                        'output_id': output_id,
                        'dataset_file': zarr_s3_path,
                        'qc_file': qc_s3_path,
                        'station': qc_metrics['file_info']['station_id'],
                        'timestamp': qc_metrics['file_info']['timestamp']
                    }
                    
                else:
                    # Local file system output
                    output_file = os.path.join(pipeline_options.output_path, f'{output_id}.zarr')
                    dataset.to_zarr(output_file, mode='w')
                    
                    # Save QC metrics
                    qc_file = os.path.join(pipeline_options.output_path, f'{output_id}_qc.json')
                    with open(qc_file, 'w') as f:
                        import json
                        json.dump(qc_metrics, f, indent=2, default=str)
                    
                    logger.info(f"Saved {output_id} -> {output_file}")
                    
                    return {
                        'output_id': output_id,
                        'dataset_file': output_file,
                        'qc_file': qc_file,
                        'station': qc_metrics['file_info']['station_id'],
                        'timestamp': qc_metrics['file_info']['timestamp']
                    }
                
            except Exception as e:
                logger.error(f"Failed to save {output_id}: {e}")
                return {'error': str(e), 'output_id': output_id}
        
        # Save results and collect summary
        results = (
            processed_data
            | 'Save datasets' >> beam.Map(save_dataset)
        )
        
        # Optional: Create summary report
        def create_summary(results_list):
            """Create processing summary."""
            summary_file = os.path.join(pipeline_options.output_path, 'processing_summary.json')
            
            summary = {
                'pipeline_options': {
                    'stations': pipeline_options.stations,
                    'start_time': pipeline_options.start_time,
                    'end_time': pipeline_options.end_time,
                    'grid_size': f"{pipeline_options.grid_size}x{pipeline_options.grid_size}",
                    'include_sweeps': pipeline_options.include_sweeps
                },
                'processing_results': {
                    'total_files_processed': len([r for r in results_list if 'error' not in r]),
                    'total_errors': len([r for r in results_list if 'error' in r]),
                    'stations_processed': list(set(r.get('station') for r in results_list if 'station' in r)),
                    'time_range': {
                        'earliest': min((r.get('timestamp') for r in results_list if 'timestamp' in r), default=None),
                        'latest': max((r.get('timestamp') for r in results_list if 'timestamp' in r), default=None)
                    }
                },
                'output_files': [r for r in results_list if 'error' not in r]
            }
            
            with open(summary_file, 'w') as f:
                import json
                json.dump(summary, f, indent=2)
            
            logger.info(f"Processing summary saved to {summary_file}")
            return summary
        
        # Collect results and create summary
        _ = (
            results
            | 'Collect results' >> beam.combiners.ToList()
            | 'Create summary' >> beam.Map(create_summary)
        )


if __name__ == '__main__':
    
    # Example usage
    pipeline_args = [
        '--stations=KABR,KPDT',
        '--start_time=2024-01-01-12:00',
        '--end_time=2024-01-01-18:00',
        '--output_path=./aws_nexrad_output',
        '--max_files_per_station=5',
        '--grid_size=80',
        '--include_sweeps=0,1,2',
        '--runner=DirectRunner',
        '--direct_running_mode=multi_processing',
        '--direct_num_workers=2'
    ]
    
    # Parse pipeline options
    pipeline_options = PipelineOptions(pipeline_args)
    nexrad_options = pipeline_options.view_as(NexradPipelineOptions)
    
    # Run pipeline
    process_nexrad_from_aws(nexrad_options)