#!/usr/bin/env python3
"""
Apache Beam pipeline for processing NEXRAD data from polar to cartesian coordinates.
Uses xarray-beam to handle zarr datasets efficiently.
"""

import os
import numpy as np
import xarray as xr
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import xarray_beam as xbeam
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelGrid:
    """Defines the target cartesian model grid."""
    
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


class QualityController:
    """Quality control and health checks for NEXRAD data."""
    
    @staticmethod
    def validate_physical_bounds(ds: xr.Dataset) -> Dict[str, bool]:
        """Validate physical bounds of radar variables."""
        checks = {}
        
        # DBZH (reflectivity) should be between -40 and 70 dBZ typically
        if 'DBZH' in ds:
            dbzh_valid = ds.DBZH.where(ds.DBZH != -33)  # -33 is missing value
            checks['dbzh_bounds'] = bool(
                (dbzh_valid >= -40).all() and (dbzh_valid <= 70).all()
            )
        
        # RHOHV (correlation coefficient) should be between 0 and 1
        if 'RHOHV' in ds:
            rhohv_valid = ds.RHOHV.where(ds.RHOHV != -33)
            checks['rhohv_bounds'] = bool(
                (rhohv_valid >= 0).all() and (rhohv_valid <= 1).all()
            )
        
        # ZDR (differential reflectivity) typically between -5 and 8 dB
        if 'ZDR' in ds:
            zdr_valid = ds.ZDR.where(ds.ZDR != -33)
            checks['zdr_bounds'] = bool(
                (zdr_valid >= -5).all() and (zdr_valid <= 8).all()
            )
            
        return checks
    
    @staticmethod
    def assess_coverage(ds: xr.Dataset) -> Dict[str, float]:
        """Assess data coverage and completeness."""
        coverage = {}
        
        total_points = ds.DBZH.size if 'DBZH' in ds else 0
        
        if total_points > 0:
            # Calculate coverage for each variable
            for var in ['DBZH', 'RHOHV', 'ZDR', 'PHIDP']:
                if var in ds:
                    valid_points = (ds[var] != -33).sum().item()
                    coverage[f'{var}_coverage'] = valid_points / total_points
        
        return coverage
    
    @staticmethod
    def signal_quality_metrics(ds: xr.Dataset) -> Dict[str, float]:
        """Calculate signal quality metrics."""
        metrics = {}
        
        if 'RHOHV' in ds:
            rhohv_valid = ds.RHOHV.where(ds.RHOHV != -33)
            metrics['mean_correlation'] = float(rhohv_valid.mean().item())
            metrics['high_correlation_fraction'] = float(
                (rhohv_valid > 0.9).sum().item() / rhohv_valid.count().item()
            )
        
        return metrics


class PolarToCartesianTransform:
    """Transform radar data from polar to cartesian coordinates."""
    
    def __init__(self, model_grid: ModelGrid):
        self.model_grid = model_grid
    
    def transform_coordinates(self, ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform polar coordinates (range, azimuth) to cartesian (lat, lon).
        
        Args:
            ds: Dataset containing radar data with range, azimuth coordinates
            
        Returns:
            Tuple of (latitude, longitude) arrays in cartesian coordinates
        """
        # Get radar location
        radar_lat = float(ds.latitude.values)
        radar_lon = float(ds.longitude.values)
        
        # Get polar coordinates
        ranges = ds.range.values  # in meters
        azimuths = ds.azimuth.values  # in degrees
        
        # Create meshgrids
        range_grid, azimuth_grid = np.meshgrid(ranges, azimuths)
        
        # Convert to radians
        azimuth_rad = np.radians(azimuth_grid)
        
        # Calculate displacement in meters
        dx = range_grid * np.sin(azimuth_rad)  # eastward displacement
        dy = range_grid * np.cos(azimuth_rad)  # northward displacement
        
        # Convert to lat/lon (approximate for small distances)
        # 1 degree latitude ≈ 111,000 meters
        # 1 degree longitude ≈ 111,000 * cos(latitude) meters
        lat_displacement = dy / 111000.0
        lon_displacement = dx / (111000.0 * np.cos(np.radians(radar_lat)))
        
        # Calculate final coordinates
        target_lats = radar_lat + lat_displacement
        target_lons = radar_lon + lon_displacement
        
        return target_lats, target_lons
    
    def regrid_to_cartesian(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Regrid polar radar data to cartesian model grid.
        
        Args:
            ds: Dataset with radar data in polar coordinates
            
        Returns:
            Dataset regridded to cartesian coordinates
        """
        # Get cartesian coordinates for radar data
        target_lats, target_lons = self.transform_coordinates(ds)
        
        # Create output grid
        x_out = self.model_grid.x_coords  # longitude
        y_out = self.model_grid.y_coords  # latitude
        
        # Create output dataset
        output_vars = {}
        
        # Process each radar variable
        radar_vars = ['DBZH', 'RHOHV', 'ZDR', 'PHIDP', 'CCORH']
        
        for var_name in radar_vars:
            if var_name in ds.data_vars:
                var_data = ds[var_name].values
                
                # Simple nearest neighbor interpolation for now
                # In production, might want more sophisticated interpolation
                gridded_data = self._interpolate_to_grid(
                    var_data, target_lats, target_lons, y_out, x_out
                )
                
                output_vars[var_name] = (('y', 'x'), gridded_data)
        
        # Create output dataset
        output_ds = xr.Dataset(
            output_vars,
            coords={
                'x': ('x', x_out),
                'y': ('y', y_out), 
                'lon': ('x', x_out),  # longitude mapping
                'lat': ('y', y_out),  # latitude mapping
            },
            attrs={
                **ds.attrs,
                'grid_type': 'cartesian',
                'original_radar_lat': float(ds.latitude.values),
                'original_radar_lon': float(ds.longitude.values),
                'regridding_method': 'nearest_neighbor'
            }
        )
        
        return output_ds
    
    def _interpolate_to_grid(self, data: np.ndarray, source_lats: np.ndarray, 
                           source_lons: np.ndarray, target_lats: np.ndarray,
                           target_lons: np.ndarray) -> np.ndarray:
        """
        Interpolate data from source to target grid using nearest neighbor.
        """
        from scipy.spatial import cKDTree
        
        # Flatten source coordinates
        source_points = np.column_stack([source_lats.ravel(), source_lons.ravel()])
        source_values = data.ravel()
        
        # Create target grid
        target_lat_grid, target_lon_grid = np.meshgrid(target_lats, target_lons, indexing='ij')
        target_points = np.column_stack([target_lat_grid.ravel(), target_lon_grid.ravel()])
        
        # Build KD-tree for efficient nearest neighbor lookup
        tree = cKDTree(source_points)
        distances, indices = tree.query(target_points)
        
        # Interpolate using nearest neighbors
        target_values = source_values[indices]
        
        # Reshape to target grid
        result = target_values.reshape(target_lat_grid.shape)
        
        # Set points too far from any data as missing
        max_distance = 0.01  # degrees, roughly 1km
        mask = distances.reshape(target_lat_grid.shape) > max_distance
        result[mask] = np.nan
        
        return result


class ProcessRadarSite(beam.DoFn):
    """Beam DoFn to process a single radar site."""
    
    def __init__(self, model_grid: ModelGrid, qc_enabled: bool = True):
        self.model_grid = model_grid
        self.qc_enabled = qc_enabled
        self.transformer = None
        self.qc = None
    
    def setup(self):
        """Initialize transformer and QC objects."""
        self.transformer = PolarToCartesianTransform(self.model_grid)
        self.qc = QualityController()
    
    def process(self, site_info: Tuple[str, str]) -> List[Tuple[str, xr.Dataset, Dict]]:
        """
        Process a single radar site.
        
        Args:
            site_info: Tuple of (site_name, zarr_path)
            
        Yields:
            Tuples of (site_name, processed_dataset, qc_metrics)
        """
        site_name, zarr_path = site_info
        
        try:
            # Process each sweep for this site
            results = []
            
            # Find available sweeps
            store = xr.open_zarr(zarr_path)
            site_group = store[site_name]
            
            sweep_groups = [k for k in site_group.group_keys() if k.startswith('sweep_')]
            
            logger.info(f"Processing site {site_name} with {len(sweep_groups)} sweeps")
            
            for sweep_name in sweep_groups[:3]:  # Process first 3 sweeps for testing
                try:
                    # Load sweep data
                    ds = xr.open_zarr(zarr_path, group=f'{site_name}/{sweep_name}')
                    
                    # Skip if no radar variables present
                    if not any(var in ds.data_vars for var in ['DBZH', 'RHOHV', 'ZDR']):
                        continue
                    
                    # Quality control
                    qc_metrics = {}
                    if self.qc_enabled:
                        qc_metrics.update(self.qc.validate_physical_bounds(ds))
                        qc_metrics.update(self.qc.assess_coverage(ds))
                        qc_metrics.update(self.qc.signal_quality_metrics(ds))
                    
                    # Transform to cartesian coordinates
                    cartesian_ds = self.transformer.regrid_to_cartesian(ds)
                    
                    # Add sweep metadata
                    cartesian_ds.attrs.update({
                        'site_name': site_name,
                        'sweep_name': sweep_name,
                        'processing_timestamp': pd.Timestamp.now().isoformat()
                    })
                    
                    results.append((f"{site_name}_{sweep_name}", cartesian_ds, qc_metrics))
                    
                except Exception as e:
                    logger.error(f"Error processing {site_name}/{sweep_name}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing site {site_name}: {e}")
            return []


def create_model_grid_from_data(zarr_path: str) -> ModelGrid:
    """Create model grid that encompasses all radar sites."""
    
    import os
    
    lats, lons = [], []
    sites = []
    
    # Check if zarr_path is a directory with individual station folders
    if os.path.isdir(zarr_path):
        # Get station directories
        for item in os.listdir(zarr_path):
            item_path = os.path.join(zarr_path, item)
            if os.path.isdir(item_path) and len(item) == 4 and item.isupper():  # Station code format
                sites.append(item)
    
    if not sites:
        # Fallback: try to open as consolidated zarr
        try:
            store = xr.open_zarr(zarr_path)
            sites = list(store.group_keys())
        except:
            # Use default sites if no data available
            sites = ['KABR', 'KPDT', 'KYUX', 'PGUA']
    
    # Get all radar locations
    for site in sites:
        try:
            site_path = os.path.join(zarr_path, site) if os.path.isdir(zarr_path) else zarr_path
            
            if os.path.isdir(site_path):
                # Try to find a sample zarr file in the site directory
                zarr_files = [f for f in os.listdir(site_path) if f.endswith('.zarr')]
                if zarr_files:
                    sample_file = os.path.join(site_path, zarr_files[0])
                    ds = xr.open_zarr(sample_file)
                else:
                    continue
            else:
                # Try hierarchical zarr
                ds = xr.open_zarr(zarr_path, group=site)
            
            if 'latitude' in ds.coords and 'longitude' in ds.coords:
                lats.append(float(ds.latitude.values))
                lons.append(float(ds.longitude.values))
        except:
            continue
    
    if not lats:
        # Return default grid if no sites found
        return ModelGrid()
    
    # Create grid with some padding around radar sites
    lat_range = max(lats) - min(lats) 
    lon_range = max(lons) - min(lons)
    
    padding_lat = max(0.5, lat_range * 0.2)  # At least 0.5 degrees padding
    padding_lon = max(0.5, lon_range * 0.2)
    
    return ModelGrid(
        nx=150, ny=150, nz=10,
        lat_min=min(lats) - padding_lat,
        lat_max=max(lats) + padding_lat,
        lon_min=min(lons) - padding_lon,
        lon_max=max(lons) + padding_lon
    )


def process_nexrad_data(input_zarr_path: str, output_path: str, 
                       pipeline_options: Optional[PipelineOptions] = None,
                       model_grid: Optional[ModelGrid] = None) -> None:
    """
    Main function to process NEXRAD data using Apache Beam.
    
    Args:
        input_zarr_path: Path to input Zarr store containing NEXRAD data
        output_path: Path where processed data should be written
        pipeline_options: Apache Beam pipeline options
        model_grid: Target cartesian grid definition
    """
    
    if pipeline_options is None:
        pipeline_options = PipelineOptions()
    
    if model_grid is None:
        model_grid = create_model_grid_from_data(input_zarr_path)
    
    logger.info(f"Processing NEXRAD data from {input_zarr_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Model grid: {model_grid.nx}x{model_grid.ny}x{model_grid.nz}")
    
    # Ensure output directory exists
    import os
    os.makedirs(output_path, exist_ok=True)
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        
        # Get list of radar sites
        sites = []
        
        if os.path.isdir(input_zarr_path):
            # Handle directory with individual station folders
            for item in os.listdir(input_zarr_path):
                item_path = os.path.join(input_zarr_path, item)
                if os.path.isdir(item_path) and len(item) == 4 and item.isupper():
                    sites.append((item, item_path))
        else:
            # Handle consolidated zarr store
            try:
                store = xr.open_zarr(input_zarr_path)
                sites = [(site, input_zarr_path) for site in store.group_keys()]
            except:
                # Fallback to default sites
                logger.warning("Could not determine sites from data, using defaults")
                default_sites = ['KABR', 'KPDT', 'KYUX', 'PGUA']
                sites = [(site, input_zarr_path) for site in default_sites]
        
        logger.info(f"Found radar sites: {[s[0] for s in sites]}")
        
        # Process sites
        processed_data = (
            pipeline
            | 'Create site list' >> beam.Create(sites)
            | 'Process radar sites' >> beam.ParDo(ProcessRadarSite(model_grid))
            | 'Flatten results' >> beam.FlatMap(lambda x: x)
        )
        
        # Save processed datasets
        def save_dataset(element):
            site_sweep_name, dataset, qc_metrics = element
            
            # Save dataset
            output_file = os.path.join(output_path, f'{site_sweep_name}.zarr')
            dataset.to_zarr(output_file, mode='w')
            
            # Save QC metrics
            qc_file = os.path.join(output_path, f'{site_sweep_name}_qc.json')
            with open(qc_file, 'w') as f:
                import json
                json.dump(qc_metrics, f, indent=2, default=str)
            
            logger.info(f"Saved {site_sweep_name} to {output_file}")
            return site_sweep_name
        
        # Save results
        _ = (
            processed_data
            | 'Save datasets' >> beam.Map(save_dataset)
        )


if __name__ == '__main__':
    # Set up pipeline options
    pipeline_options = PipelineOptions([
        '--runner=DirectRunner',  # Use direct runner for local execution
        '--direct_running_mode=multi_processing',
        '--direct_num_workers=2'
    ])
    
    # Run the pipeline
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data')
    output_path = os.path.join(base_dir, 'out')
    
    print(f"Looking for data in: {data_path}")
    print(f"Output will be saved to: {output_path}")
    
    process_nexrad_data(
        input_zarr_path=data_path,
        output_path=output_path,
        pipeline_options=pipeline_options
    )
