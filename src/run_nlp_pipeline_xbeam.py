#!/usr/bin/env python3
"""
Natural Language NEXRAD Processing Pipeline using xarray_beam.
Demonstrates proper integration between xarray.Dataset and Apache Beam Pipeline.
"""

import os
import sys
import argparse
import logging
import json
import re
import tempfile
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import xarray_beam as xbeam
import xarray as xr
import numpy as np
import boto3
from botocore.config import Config
import anthropic
import xradar as xd

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Parsed natural language query result."""
    stations: List[str]
    start_time: datetime
    end_time: datetime
    confidence: float = 0.0


@dataclass
class ModelGrid:
    """Defines the target cartesian model grid for coordinate transformation."""
    
    # Grid dimensions
    nx: int = 100  # x-direction grid points
    ny: int = 100  # y-direction grid points  
    nz: int = 10   # z-direction grid points (elevation levels)
    
    # Spatial bounds (in degrees for lat/lon)
    lat_min: float = 39.0
    lat_max: float = 41.0
    lon_min: float = -97.0
    lon_max: float = -95.0
    
    # Elevation levels (in meters)
    elevation_levels: List[float] = None
    
    def __post_init__(self):
        if self.elevation_levels is None:
            # Default elevation levels from 0 to 10km
            self.elevation_levels = [i * 1000.0 for i in range(self.nz)]
    
    @property
    def x_coords(self) -> np.ndarray:
        """X coordinates (longitude)."""
        return np.linspace(self.lon_min, self.lon_max, self.nx)
    
    @property 
    def y_coords(self) -> np.ndarray:
        """Y coordinates (latitude)."""
        return np.linspace(self.lat_min, self.lat_max, self.ny)
    
    @property
    def z_coords(self) -> np.ndarray:
        """Z coordinates (elevation)."""
        return np.array(self.elevation_levels)


class PolarToCartesianTransform:
    """Transform radar data from polar to cartesian coordinates."""
    
    def __init__(self, model_grid: ModelGrid):
        self.model_grid = model_grid
    
    def transform_coordinates(self, ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Transform polar coordinates (range, azimuth) to cartesian (lat, lon)."""
        # Check if dataset has necessary coordinates
        if 'range' not in ds.dims or 'azimuth' not in ds.dims:
            logger.warning("Dataset missing range or azimuth dimensions for polar transformation")
            return None, None
            
        # Get radar location (use defaults if not available)
        radar_lat = float(ds.attrs.get('latitude', 40.0))
        radar_lon = float(ds.attrs.get('longitude', -96.0))
        
        # Get polar coordinates
        ranges = ds.range.values if 'range' in ds else np.array([1000])  # in meters
        azimuths = ds.azimuth.values if 'azimuth' in ds else np.array([0])  # in degrees
        
        # Create meshgrids
        range_grid, azimuth_grid = np.meshgrid(ranges, azimuths)
        
        # Convert to radians
        azimuth_rad = np.radians(azimuth_grid)
        
        # Calculate displacement in meters
        dx = range_grid * np.sin(azimuth_rad)  # eastward displacement
        dy = range_grid * np.cos(azimuth_rad)  # northward displacement
        
        # Convert to lat/lon (approximate for small distances)
        lat_displacement = dy / 111000.0
        lon_displacement = dx / (111000.0 * np.cos(np.radians(radar_lat)))
        
        # Calculate final coordinates
        target_lats = radar_lat + lat_displacement
        target_lons = radar_lon + lon_displacement
        
        return target_lats, target_lons
    
    def regrid_to_cartesian(self, ds: xr.Dataset) -> xr.Dataset:
        """Regrid polar radar data to cartesian model grid."""
        # Get cartesian coordinates for radar data
        target_lats, target_lons = self.transform_coordinates(ds)
        
        if target_lats is None or target_lons is None:
            # Return original dataset if transformation not possible
            logger.warning("Cannot transform coordinates, returning original dataset")
            return ds
        
        # Create output grid
        x_out = self.model_grid.x_coords  # longitude
        y_out = self.model_grid.y_coords  # latitude
        
        # Create output dataset with transformed coordinates
        output_ds = xr.Dataset(
            coords={
                'latitude': (['y', 'x'], np.tile(y_out[:, np.newaxis], (1, len(x_out)))),
                'longitude': (['y', 'x'], np.tile(x_out[np.newaxis, :], (len(y_out), 1))),
                'x': x_out,
                'y': y_out
            },
            attrs={
                **ds.attrs,
                'coordinate_system': 'cartesian',
                'transformed': True,
                'grid_nx': self.model_grid.nx,
                'grid_ny': self.model_grid.ny
            }
        )
        
        # Interpolate data variables to new grid
        from scipy.interpolate import griddata
        
        # Flatten polar coordinates for interpolation
        points = np.column_stack((target_lons.flatten(), target_lats.flatten()))
        
        # Create target grid points
        xi, yi = np.meshgrid(x_out, y_out)
        target_points = np.column_stack((xi.flatten(), yi.flatten()))
        
        # Copy and interpolate data variables
        for var in ds.data_vars:
            if var not in ['range', 'azimuth'] and ds[var].dims == ('azimuth', 'range'):
                # Get data values
                data = ds[var].values
                values = data.flatten()
                
                # Remove NaN values for interpolation
                mask = ~np.isnan(values)
                if np.any(mask):
                    # Interpolate to cartesian grid
                    interpolated = griddata(
                        points[mask], 
                        values[mask], 
                        target_points, 
                        method='linear',
                        fill_value=np.nan
                    )
                    interpolated = interpolated.reshape((len(y_out), len(x_out)))
                else:
                    # If no valid data, fill with NaN
                    interpolated = np.full((len(y_out), len(x_out)), np.nan)
                
                output_ds[var] = xr.DataArray(
                    data=interpolated,
                    dims=['y', 'x'],
                    coords={'y': y_out, 'x': x_out},
                    attrs=ds[var].attrs if var in ds else {}
                )
        
        logger.debug(f"Transformed dataset to cartesian grid ({self.model_grid.nx}x{self.model_grid.ny})")
        return output_ds


class SimpleNLParser:
    """Claude API-powered natural language parser for NEXRAD queries."""
    
    def __init__(self):
        # Initialize Claude client
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def parse_query(self, query: str) -> ParsedQuery:
        """Parse natural language query using Claude API."""
        
        system_prompt = """You are a NEXRAD weather radar data query parser. Your task is to parse natural language queries into structured parameters for radar data processing.

Return a JSON object with these fields:
- stations: Array of 4-letter radar station codes (e.g., ["KABR", "KPDT"]) or null for all stations
- start_time: ISO datetime string (YYYY-MM-DDTHH:MM:SS) 
- end_time: ISO datetime string (YYYY-MM-DDTHH:MM:SS)
- confidence: Float between 0-1 indicating parsing confidence

Common radar stations: KABR, KPDT, KYUX, KSGF, KRTX, PGUA, KLWX, KJAX, KDIX, KBGM

Time parsing rules:
- "yesterday" = previous day 12:00-18:00 unless times specified
- "last N hours" = N hours ago to now
- Specific dates like "November 1st 2024" = that date 12:00-18:00 unless times specified
- Default fallback: November 1st 2024 00:00-06:00 (known to have data)

Examples:
- "Get KABR radar data from yesterday 2pm to 6pm" -> stations: ["KABR"], yesterday 14:00-18:00
- "Process all stations from last 3 hours" -> stations: null, 3 hours ago to now
- "KYUX data from November 1st 2024" -> stations: ["KYUX"], 2024-11-01 12:00-18:00

Return only the JSON object, no additional text."""
        
        try:
            # Call Claude API
            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Parse this NEXRAD query: {query}"}
                ]
            )
            
            # Parse the JSON response
            response_text = response.content[0].text.strip()
            
            # Clean up response if it has markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif response_text.startswith('```'):
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            parsed_response = json.loads(response_text)
            
            # Convert to ParsedQuery object
            stations = parsed_response.get('stations')
            start_time_str = parsed_response.get('start_time')
            end_time_str = parsed_response.get('end_time')
            confidence = parsed_response.get('confidence', 0.8)
            
            # Parse datetime strings
            start_time = datetime.fromisoformat(start_time_str)
            end_time = datetime.fromisoformat(end_time_str)
            
            logger.info(f"Claude API parsed query successfully: stations={stations}, "
                       f"time_range={start_time} to {end_time}, confidence={confidence}")
            
            return ParsedQuery(
                stations=stations,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"Claude API parsing failed: {e}. Using default parameters for entire dataset.")
            return self._get_default_params()
    
    def _get_default_params(self) -> ParsedQuery:
        """Return default parameters to process entire dataset."""
        
        # Default to processing entire available dataset
        base_date = datetime(2024, 11, 1)  # Known date with data
        start_time = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = base_date.replace(hour=6, minute=0, second=0, microsecond=0)
        
        logger.info(f"Using default parameters: all stations, "
                   f"time_range={start_time} to {end_time}")
            
        return ParsedQuery(
            stations=None,  # Process all available stations
            start_time=start_time,
            end_time=end_time,
            confidence=0.3  # Low confidence for defaults
        )


class NexradPipelineOptions(PipelineOptions):
    """Custom pipeline options for NEXRAD processing."""
    
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument('--stations', dest='stations', default='')
        parser.add_argument('--start_time', dest='start_time', required=False, default='')
        parser.add_argument('--end_time', dest='end_time', required=False, default='')
        parser.add_argument('--output_path', dest='output_path', default='./out')
        parser.add_argument('--max_files_per_station', dest='max_files_per_station', type=int, default=3)
        parser.add_argument('--max_stations', dest='max_stations', type=int, default=10)
        parser.add_argument('--rechunk_size', dest='rechunk_size', default='auto')


class NexradFileInfo:
    """Information about a NEXRAD file to process."""
    def __init__(self, station: str, s3_key: str, timestamp: datetime):
        self.station = station
        self.s3_key = s3_key
        self.timestamp = timestamp
        self.local_path = None


def download_nexrad_file(file_info: NexradFileInfo) -> Optional[str]:
    """Download NEXRAD file from S3 and return local path."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as tmp_file:
            # Download from S3
            s3_client = boto3.client('s3', region_name='us-east-1')
            s3_client.download_file('unidata-nexrad-level2', file_info.s3_key, tmp_file.name)
            
            file_info.local_path = tmp_file.name
            logger.info(f"Downloaded {file_info.station} file: {os.path.basename(file_info.s3_key)}")
            return tmp_file.name
            
    except Exception as e:
        logger.error(f"Failed to download {file_info.s3_key}: {e}")
        return None


def load_nexrad_as_xarray(file_info: NexradFileInfo) -> Optional[xr.Dataset]:
    """Load NEXRAD file as xarray Dataset using xradar."""
    if not file_info.local_path or not os.path.exists(file_info.local_path):
        local_path = download_nexrad_file(file_info)
        if not local_path:
            return None
    else:
        local_path = file_info.local_path
    
    try:
        # Use xradar to open NEXRAD Level 2 data
        try:
            # Open with xradar which handles NEXRAD Level 2 format
            dtree = xd.io.open_nexradlevel2_datatree(local_path)
            
            # Get the first sweep (you can iterate through all if needed)
            # NEXRAD data is organized in sweeps (elevation angles)
            sweep_keys = [k for k in dtree.keys() if k.startswith('sweep_')]
            
            if sweep_keys:
                # Get first sweep for now (can be extended to process all sweeps)
                first_sweep = dtree[sweep_keys[0]].to_dataset()
                
                # xradar provides data in polar coordinates (range, azimuth)
                logger.info(f"Loaded {file_info.station} with xradar - dims: {dict(first_sweep.dims)}")
                
                # Add metadata
                first_sweep.attrs.update({
                    'station': file_info.station,
                    'source_s3_key': file_info.s3_key,
                    'timestamp': file_info.timestamp.isoformat(),
                    'local_file': local_path,
                    'sweep_count': len(sweep_keys),
                    'current_sweep': sweep_keys[0]
                })
                
                # Add radar location if available
                if 'latitude' in dtree.attrs:
                    first_sweep.attrs['latitude'] = dtree.attrs['latitude']
                if 'longitude' in dtree.attrs:
                    first_sweep.attrs['longitude'] = dtree.attrs['longitude']
                if 'altitude' in dtree.attrs:
                    first_sweep.attrs['altitude'] = dtree.attrs['altitude']
                
                return first_sweep
            else:
                logger.warning(f"No sweep data found in {file_info.s3_key}")
                return None
                
        except Exception as e:
            logger.warning(f"xradar failed to read {file_info.s3_key}: {e}, trying fallback")
            # Fallback to basic dataset if xradar fails
            ds = xr.Dataset(
                data_vars={
                    'station': (['time'], [file_info.station]),
                    'file_path': (['time'], [file_info.s3_key]),
                    'file_size': (['time'], [os.path.getsize(local_path)])
                },
                coords={
                    'time': [file_info.timestamp]
                },
                attrs={
                    'station': file_info.station,
                    'source_file': file_info.s3_key,
                    'processing_time': datetime.now().isoformat()
                }
            )
            logger.debug(f"Created minimal dataset for {file_info.station}")
            return ds
        
    except Exception as e:
        logger.error(f"Failed to load {file_info.s3_key} as xarray: {e}")
        return None
    finally:
        # Clean up temporary file
        try:
            if local_path and os.path.exists(local_path):
                os.unlink(local_path)
        except:
            pass


def process_nexrad_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Process a NEXRAD dataset - apply quality control and transformations."""
    
    # Add processing metadata
    processed_ds = ds.copy()
    processed_ds.attrs['processed'] = True
    processed_ds.attrs['processing_timestamp'] = datetime.now().isoformat()
    
    # Simple quality control - add QC flags
    if 'time' in processed_ds.dims:
        # Add quality control variables
        qc_flags = xr.DataArray(
            data=np.ones(processed_ds.sizes['time'], dtype=np.int32),
            dims=['time'],
            coords={'time': processed_ds['time']},
            name='qc_flags',
            attrs={'description': 'Quality control flags: 1=good, 0=bad'}
        )
        processed_ds['qc_flags'] = qc_flags
    
    # Add processing statistics
    processed_ds.attrs['n_time_steps'] = processed_ds.sizes.get('time', 0)
    processed_ds.attrs['data_variables'] = list(processed_ds.data_vars.keys())
    
    # Add chunk information for debugging
    chunk_info = {}
    for var_name, var in processed_ds.data_vars.items():
        if hasattr(var.data, 'chunks'):
            chunk_info[var_name] = str(var.data.chunks)
    processed_ds.attrs['chunk_info'] = str(chunk_info)
    
    logger.debug(f"Processed dataset for station {processed_ds.attrs.get('station', 'unknown')}")
    return processed_ds


class CreateNexradFileInfos(beam.DoFn):
    """Create NexradFileInfo objects from file discovery."""
    
    def process(self, file_tuple: Tuple[str, str, datetime]):
        """Convert file tuple to NexradFileInfo object."""
        station, s3_key, timestamp = file_tuple
        yield NexradFileInfo(station, s3_key, timestamp)


class LoadNexradDataset(beam.DoFn):
    """Load NEXRAD files as xarray Datasets using xarray_beam patterns."""
    
    def __init__(self, is_local_data: bool = False):
        self.is_local_data = is_local_data
    
    def process(self, file_info: NexradFileInfo) -> Iterator[xr.Dataset]:
        """Load NEXRAD file as xarray Dataset."""
        if self.is_local_data:
            ds = self.load_local_dataset(file_info)
        else:
            ds = load_nexrad_as_xarray(file_info)
            
        if ds is not None:
            yield ds
    
    def load_local_dataset(self, file_info: NexradFileInfo) -> Optional[xr.Dataset]:
        """Load local zarr dataset from consolidated store."""
        try:
            data_dir = file_info.s3_key  # For local data, this contains the data directory path
            station = file_info.station
            
            # For consolidated zarr stores, open the specific station group
            if os.path.exists(os.path.join(data_dir, 'zarr.json')):
                # Open consolidated zarr with the specific station group
                ds = xr.open_zarr(data_dir, group=station, consolidated=True)
                logger.info(f"Loaded station {station} from consolidated zarr store")
            elif data_dir.endswith('.zarr'):
                ds = xr.open_zarr(data_dir)
                logger.info(f"Loaded local zarr dataset: {os.path.basename(data_dir)}")
            elif data_dir.endswith('.nc'):
                ds = xr.open_dataset(data_dir)
                logger.info(f"Loaded local netcdf dataset: {os.path.basename(data_dir)}")
            else:
                logger.warning(f"Unsupported local file format: {data_dir}")
                return None
            
            # Add metadata for local files
            ds.attrs.update({
                'station': file_info.station,
                'source_file': data_dir,
                'timestamp': file_info.timestamp.isoformat(),
                'local_processing': True
            })
            
            # Extract radar location from dataset if available
            if 'latitude' in ds.variables:
                ds.attrs['latitude'] = float(ds.latitude.values)
            if 'longitude' in ds.variables:
                ds.attrs['longitude'] = float(ds.longitude.values)
            
            return ds
            
        except Exception as e:
            logger.error(f"Failed to load local dataset for station {file_info.station}: {e}")
            return None


class ProcessDataset(beam.DoFn):
    """Process xarray Datasets with coordinate transformation and rechunking support."""
    
    def __init__(self, model_grid: ModelGrid = None, rechunk_size: str = 'auto'):
        self.rechunk_size = rechunk_size
        self.model_grid = model_grid or ModelGrid()  # Use default grid if not provided
        self.transformer = None
    
    def setup(self):
        """Initialize the coordinate transformer."""
        self.transformer = PolarToCartesianTransform(self.model_grid)
    
    def process(self, ds: xr.Dataset) -> Iterator[xr.Dataset]:
        """Apply processing to dataset including coordinate transformation."""
        # First apply basic processing
        processed_ds = process_nexrad_dataset(ds)
        
        # Apply coordinate transformation from polar to cartesian
        if self.transformer:
            try:
                transformed_ds = self.transformer.regrid_to_cartesian(processed_ds)
                logger.debug(f"Applied polar to cartesian transformation for station {processed_ds.attrs.get('station', 'unknown')}")
                processed_ds = transformed_ds
            except Exception as e:
                logger.warning(f"Coordinate transformation failed: {e}, using original coordinates")
        else:
            logger.debug("No transformer available, skipping coordinate transformation")
        
        # Apply rechunking if specified
        if self.rechunk_size != 'auto':
            try:
                rechunked_ds = self._rechunk_dataset(processed_ds)
                logger.debug(f"Rechunked dataset for station {processed_ds.attrs.get('station', 'unknown')}")
                yield rechunked_ds
            except Exception as e:
                logger.warning(f"Rechunking failed: {e}, using original chunks")
                yield processed_ds
        else:
            yield processed_ds
    
    def _rechunk_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Rechunk dataset according to specified strategy."""
        
        # Parse rechunk size
        if self.rechunk_size.isdigit():
            # Simple numeric chunk size
            chunk_dict = {'time': int(self.rechunk_size)}
        elif self.rechunk_size == 'optimal':
            # Optimal chunking for NEXRAD data
            chunk_dict = self._get_optimal_chunks(ds)
        elif ',' in self.rechunk_size:
            # Parse dimension-specific chunks: "time:100,range:500"
            chunk_dict = {}
            for chunk_spec in self.rechunk_size.split(','):
                if ':' in chunk_spec:
                    dim, size = chunk_spec.split(':')
                    chunk_dict[dim.strip()] = int(size)
        else:
            # Default to time-based chunking
            chunk_dict = {'time': 1}
        
        # Apply rechunking only to dimensions that exist
        valid_chunks = {dim: size for dim, size in chunk_dict.items() 
                       if dim in ds.dims}
        
        if valid_chunks:
            return ds.chunk(valid_chunks)
        else:
            return ds
    
    def _get_optimal_chunks(self, ds: xr.Dataset) -> Dict[str, int]:
        """Calculate optimal chunk sizes for NEXRAD data."""
        optimal_chunks = {}
        
        # Time dimension: small chunks for temporal processing
        if 'time' in ds.dims:
            time_size = ds.sizes['time']
            optimal_chunks['time'] = min(10, time_size)  # Max 10 time steps per chunk
        
        # Spatial dimensions: larger chunks for spatial operations
        spatial_dims = ['range', 'azimuth', 'x', 'y', 'lat', 'lon']
        for dim in spatial_dims:
            if dim in ds.dims:
                dim_size = ds.sizes[dim]
                if dim_size <= 100:
                    optimal_chunks[dim] = dim_size  # Keep small dimensions unchunked
                elif dim_size <= 1000:
                    optimal_chunks[dim] = dim_size // 2  # Split large dimensions
                else:
                    optimal_chunks[dim] = 500  # Large chunks for very big dimensions
        
        return optimal_chunks


class SaveDatasetToOutput(beam.DoFn):
    """Save processed datasets to output location."""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.is_s3_output = output_path.startswith('s3://')
    
    def process(self, ds: xr.Dataset) -> Iterator[Dict[str, str]]:
        """Save dataset to output location."""
        station = ds.attrs.get('station', 'unknown')
        timestamp = ds.attrs.get('timestamp', datetime.now().isoformat())
        
        try:
            # Parse timestamp for filename
            if isinstance(timestamp, str):
                ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                ts = timestamp
                
            filename = f"{ts.strftime('%Y%m%d_%H%M')}.zarr"
            
            if self.is_s3_output:
                # S3 output
                output_path = f"{self.output_path.rstrip('/')}/{station}/{filename}"
                ds.to_zarr(output_path, mode='w')
                logger.info(f"✓ Saved {station} dataset to S3: {output_path}")
                
            else:
                # Local output
                station_dir = os.path.join(self.output_path, station)
                os.makedirs(station_dir, exist_ok=True)
                output_path = os.path.join(station_dir, filename)
                ds.to_zarr(output_path, mode='w')
                logger.info(f"✓ Saved {station} dataset locally: {output_path}")
            
            yield {
                'station': str(station),
                'output_path': str(output_path),
                'timestamp': str(timestamp),
                'n_variables': str(len(ds.data_vars)),
                'dims': str(dict(ds.sizes))
            }
            
        except Exception as e:
            logger.error(f"Failed to save dataset for {station}: {e}")
            # Yield error info instead of failing completely
            yield {
                'station': str(station),
                'error': str(e),
                'timestamp': str(timestamp)
            }


def find_local_data_files(data_dir: str = 'data/') -> List[tuple]:
    """Find NEXRAD stations in local consolidated zarr store."""
    import zarr
    
    logger.info(f"Searching for local data in {data_dir}")
    
    all_files = []
    
    # Data is guaranteed to be a consolidated zarr store
    try:
        # Open consolidated zarr store
        store = zarr.open_consolidated(data_dir)
        # Get all groups (stations)
        for station_name, group in store.groups():
            logger.info(f"Found station {station_name} in consolidated zarr store")
            # Use the data directory path as the "file path" for this station
            timestamp = datetime.now()
            all_files.append((station_name, data_dir, timestamp))
        logger.info(f"Found {len(all_files)} stations in consolidated zarr store")
    except Exception as e:
        logger.error(f"Failed to open consolidated zarr store: {e}")
                
    return all_files


def discover_available_stations(date: datetime) -> List[str]:
    """Discover all available radar stations for a given date from S3."""
    
    s3_client = boto3.client('s3', region_name='us-east-1')
    date_prefix = f"{date.strftime('%Y/%m/%d')}/"
    
    logger.info(f"Discovering available stations for {date.date()}")
    
    try:
        # List all prefixes (subdirectories) under the date
        response = s3_client.list_objects_v2(
            Bucket='unidata-nexrad-level2',
            Prefix=date_prefix,
            Delimiter='/'
        )
        
        stations = []
        for prefix_info in response.get('CommonPrefixes', []):
            prefix = prefix_info['Prefix']
            # Extract station name: "2024/11/01/KABR/" -> "KABR"
            station = prefix.replace(date_prefix, '').rstrip('/')
            if len(station) == 4 and station.isupper():  # Valid station format
                stations.append(station)
        
        stations.sort()
        logger.info(f"Discovered {len(stations)} stations: {stations[:10]}..." if len(stations) > 10 else f"Discovered {len(stations)} stations: {stations}")
        return stations
        
    except Exception as e:
        logger.warning(f"Failed to discover stations for {date.date()}: {e}")
        # Fallback to common stations
        return ['KABR', 'KPDT', 'KYUX', 'KSGF', 'KRTX', 'PGUA']


def find_nexrad_files(stations: Optional[List[str]], start_time: datetime, end_time: datetime, 
                      max_files_per_station: int = 3, max_stations: int = 10) -> List[tuple]:
    """Find NEXRAD files in the AWS bucket."""
    
    # If no stations specified, discover available stations
    if stations is None:
        all_stations = discover_available_stations(start_time)
        # Limit the number of stations to process for efficiency
        stations = all_stations[:max_stations]
        logger.info(f"Discovered {len(all_stations)} stations, processing first {len(stations)}: {stations}")
    
    logger.info(f"Searching for files from {len(stations)} stations: {stations[:5]}..." if len(stations) > 5 else f"Searching for files from stations {stations}")
    logger.info(f"Time range: {start_time} to {end_time}")
    
    # Use unsigned access for public NEXRAD bucket
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    all_files = []
    
    for station in stations:
        station_files = []
        
        # Generate date range
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            prefix = f"{current_date.strftime('%Y/%m/%d')}/{station}/"
            
            try:
                paginator = s3_client.get_paginator('list_objects_v2')
                
                for page in paginator.paginate(Bucket='unidata-nexrad-level2', Prefix=prefix):
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        
                        # Extract timestamp from filename
                        filename = os.path.basename(key)
                        if filename.startswith(station):
                            try:
                                # Parse timestamp from filename: KABR20241101_000113_V06
                                parts = filename.split('_')
                                if len(parts) >= 2:
                                    date_time_str = parts[0][4:] + parts[1]  # 20241101000113
                                    file_datetime = datetime.strptime(date_time_str, '%Y%m%d%H%M%S')
                                    
                                    if start_time <= file_datetime <= end_time:
                                        station_files.append((station, key, file_datetime))
                                    
                            except:
                                continue
                                
            except Exception as e:
                logger.warning(f"Error listing files for {station} on {current_date}: {e}")
                
            current_date += timedelta(days=1)
        
        # Sort by time and limit
        station_files.sort(key=lambda x: x[2])
        station_files = station_files[:max_files_per_station]
        
        logger.info(f"Found {len(station_files)} files for {station}")
        all_files.extend(station_files)
    
    logger.info(f"Total files to process: {len(all_files)}")
    return all_files


def run_xarray_beam_pipeline(options: NexradPipelineOptions):
    """Run the NEXRAD processing pipeline using xarray_beam patterns."""
    
    stations = options.stations.split(',') if options.stations else None
    start_time = datetime.strptime(options.start_time, '%Y-%m-%d-%H:%M') if options.start_time else datetime.now()
    end_time = datetime.strptime(options.end_time, '%Y-%m-%d-%H:%M') if options.end_time else datetime.now()
    
    # Find files to process
    files_to_process = find_nexrad_files(stations, start_time, end_time, 
                                       options.max_files_per_station,
                                       options.max_stations)
    
    if not files_to_process:
        logger.error("No files found for the specified criteria")
        return
    
    logger.info(f"Processing {len(files_to_process)} files with xarray_beam pipeline")
    
    # Create output directory (only for local paths)
    if not options.output_path.startswith('s3://'):
        os.makedirs(options.output_path, exist_ok=True)
    else:
        logger.info(f"Output will be saved to S3: {options.output_path}")
    
    # Build the xarray_beam pipeline
    with beam.Pipeline(options=options) as pipeline:
        
        # Step 1: Create file information objects
        file_infos = (
            pipeline
            | 'Create file tuples' >> beam.Create(files_to_process)
            | 'Create file info objects' >> beam.ParDo(CreateNexradFileInfos())
        )
        
        # Step 2: Load files as xarray Datasets
        datasets = (
            file_infos
            | 'Load as xarray Datasets' >> beam.ParDo(LoadNexradDataset(is_local_data=False))
        )
        
        # Step 3: Process datasets (quality control, transformations, rechunking, coordinate transformation)
        # Create model grid for coordinate transformation
        model_grid = ModelGrid()
        processed_datasets = (
            datasets
            | 'Process datasets' >> beam.ParDo(ProcessDataset(model_grid=model_grid, rechunk_size=options.rechunk_size))
        )
        
        # Step 4: Save processed datasets
        results = (
            processed_datasets
            | 'Save to output' >> beam.ParDo(SaveDatasetToOutput(options.output_path))
        )
        
        # Step 5: Log results
        _ = (
            results
            | 'Log results' >> beam.Map(lambda x: logger.info(f"Pipeline result: {x}"))
        )
    
    logger.info("✅ xarray_beam pipeline completed successfully!")
    logger.info(f"Results saved to: {options.output_path}")


def run_local_data_pipeline(options: PipelineOptions, rechunk_size: str, output_path: str):
    """Run the NEXRAD processing pipeline for local data files."""
    
    # Find local data files
    files_to_process = find_local_data_files('data/')
    
    if not files_to_process:
        logger.error("No local data files found in data/ directory")
        return
    
    logger.info(f"Processing {len(files_to_process)} local files with xarray_beam pipeline")
    
    # Create output directory (only for local paths)
    if not output_path.startswith('s3://'):
        os.makedirs(output_path, exist_ok=True)
    else:
        logger.info(f"Output will be saved to S3: {output_path}")
    
    # Build the xarray_beam pipeline for local data
    with beam.Pipeline(options=options) as pipeline:
        
        # Step 1: Create file information objects
        file_infos = (
            pipeline
            | 'Create local file tuples' >> beam.Create(files_to_process)
            | 'Create file info objects' >> beam.ParDo(CreateNexradFileInfos())
        )
        
        # Step 2: Load local files as xarray Datasets
        datasets = (
            file_infos
            | 'Load local xarray Datasets' >> beam.ParDo(LoadNexradDataset(is_local_data=True))
        )
        
        # Step 3: Process datasets (quality control, transformations, rechunking, coordinate transformation)
        model_grid = ModelGrid()
        processed_datasets = (
            datasets
            | 'Process local datasets' >> beam.ParDo(ProcessDataset(model_grid=model_grid, rechunk_size=rechunk_size))
        )
        
        # Step 4: Save processed datasets
        results = (
            processed_datasets
            | 'Save local data to output' >> beam.ParDo(SaveDatasetToOutput(output_path))
        )
        
        # Step 5: Log results
        _ = (
            results
            | 'Log local results' >> beam.Map(lambda x: logger.info(f"Local pipeline result: {x}"))
        )
    
    logger.info("✅ Local data xarray_beam pipeline completed successfully!")
    logger.info(f"Results saved to: {output_path}")


def validate_s3_bucket(bucket_name: str) -> bool:
    """Validate S3 bucket name format."""
    import re
    
    # Basic S3 bucket name validation
    if not bucket_name:
        return False
    
    # Remove s3:// prefix if present
    if bucket_name.startswith('s3://'):
        bucket_name = bucket_name[5:].split('/')[0]
    
    # Check bucket name rules
    if not (3 <= len(bucket_name) <= 63):
        return False
    
    # Must start and end with lowercase letter or number
    if not re.match(r'^[a-z0-9].*[a-z0-9]$', bucket_name):
        return False
    
    # No uppercase, underscores, or consecutive periods
    if re.search(r'[A-Z_]|\.\.', bucket_name):
        return False
    
    return True


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description='Natural Language NEXRAD Pipeline with xarray_beam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process local data directory (default behavior)
    python run_nlp_pipeline_xbeam.py
    
    # Query for specific station data from NEXRAD S3
    python run_nlp_pipeline_xbeam.py --query "Process KABR radar data from November 1st 2024"
    
    # Query for all stations data from NEXRAD S3
    python run_nlp_pipeline_xbeam.py --query "Process radar data from November 1st 2024"
    
    # S3 output
    python run_nlp_pipeline_xbeam.py --query "Get data from yesterday" --output s3://my-bucket/results
    
    # Local data to custom local path
    python run_nlp_pipeline_xbeam.py --output /custom/output/path
        """
    )
    parser.add_argument('--query', help='Natural language query for live NEXRAD data processing. If not specified, processes local data/ directory')
    parser.add_argument('--output', default='out/', help='Output directory or S3 path (e.g., s3://my-bucket/results)')
    parser.add_argument('--dry_run', action='store_true', help='Parse only, do not run')
    parser.add_argument('--rechunk_size', default='auto', 
                       help='Rechunk strategy: "auto" (no rechunking), "optimal" (smart chunking), '
                            '"N" (time chunks of size N), or "dim:size,dim:size" (e.g., "time:10,range:500")')
    
    args, beam_args = parser.parse_known_args()
    
    # Determine processing mode
    logger.info(f"🧠 Natural Language NEXRAD Pipeline with xarray_beam")
    
    # Determine output path and validate S3 if needed
    output_path = args.output
    if output_path.startswith('s3://'):
        # Validate S3 path
        if not validate_s3_bucket(output_path):
            logger.error(f"Invalid S3 output path: {output_path}")
            return 1
        logger.info(f"Output: S3 path {output_path}")
    else:
        logger.info(f"Output: Local directory {output_path}")
    
    if args.query:
        # Natural language query mode for live data
        logger.info(f"Mode: Live data processing")
        logger.info(f"Query: '{args.query}'")
        
        nlp = SimpleNLParser()
        parsed = nlp.parse_query(args.query)
        
        logger.info(f"✓ Parsed query (confidence: {parsed.confidence:.2f})")
        logger.info(f"  Stations: {parsed.stations}")
        logger.info(f"  Time range: {parsed.start_time} to {parsed.end_time}")
        
        # Use parsed query parameters
        stations = parsed.stations
        start_time = parsed.start_time
        end_time = parsed.end_time
        
    else:
        # Local data processing mode
        logger.info(f"Mode: Local data directory processing")
        logger.info(f"Input: data/ directory")
        
        # Default parameters for local data processing
        stations = None  # Will use all available stations from data/ directory
        start_time = None
        end_time = None
    
    logger.info(f"  Using xarray_beam for dataset processing")
    logger.info(f"  Rechunking strategy: {args.rechunk_size}")
    
    if args.dry_run:
        logger.info("🏃 Dry run mode - configuration validated successfully")
        return 0
    
    # Create beam arguments
    if args.query:
        # Live data processing
        stations_arg = ",".join(stations) if stations else ""
        beam_args.extend([
            f'--stations={stations_arg}',
            f'--start_time={start_time.strftime("%Y-%m-%d-%H:%M")}',
            f'--end_time={end_time.strftime("%Y-%m-%d-%H:%M")}',
            f'--output_path={output_path}',
            f'--rechunk_size={args.rechunk_size}',
            '--runner=DirectRunner',
            '--direct_num_workers=1'
        ])
        
        logger.info("🚀 Starting NEXRAD processing pipeline with live data...")
        pipeline_options = PipelineOptions(beam_args)
        nexrad_options = pipeline_options.view_as(NexradPipelineOptions)
        run_xarray_beam_pipeline(nexrad_options)
        
    else:
        # Local data processing
        beam_args.extend([
            f'--output_path={output_path}',
            f'--rechunk_size={args.rechunk_size}',
            '--runner=DirectRunner',
            '--direct_num_workers=1'
        ])
        
        logger.info("🚀 Starting NEXRAD processing pipeline with local data...")
        pipeline_options = PipelineOptions(beam_args)
        run_local_data_pipeline(pipeline_options, args.rechunk_size, output_path)
    
    # Add helpful S3 console link if using S3 output
    if output_path.startswith('s3://'):
        bucket_name = output_path.replace('s3://', '').split('/')[0]
        logger.info(f"View results at: https://s3.console.aws.amazon.com/s3/buckets/{bucket_name}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())