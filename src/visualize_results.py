#!/usr/bin/env python3
"""
Create visualizations to demonstrate the NEXRAD processing results.
"""

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import logging
import zarr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_comparison_plots(data_path: str, output_path: str, save_path: str = '../visualizations', transformed_path: str = None):
    """Create before/after comparison plots."""
    
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Read data from both directories
        original_data = {}
        processed_data = {}
        
        # Read original data if it exists
        if os.path.exists(data_path):
            if os.path.exists(os.path.join(data_path, 'zarr.json')):
                # Consolidated zarr store
                logger.info("Reading original data from consolidated zarr store")
                store = zarr.open_consolidated(data_path)
                for station_name, _ in store.groups():
                    original_data[station_name] = xr.open_zarr(data_path, group=station_name, consolidated=True)
                    logger.info(f"  Found original data for station {station_name}")
        
        # Read processed data from both output paths
        paths_to_check = [output_path]
        if transformed_path and os.path.exists(transformed_path):
            paths_to_check.append(transformed_path)
        
        for path in paths_to_check:
            if os.path.exists(path):
                logger.info(f"Reading processed data from: {path}")
                for station_dir in os.listdir(path):
                    station_path = os.path.join(path, station_dir)
                    if os.path.isdir(station_path):
                        # Look for .zarr directories (not files)
                        zarr_dirs = [f for f in os.listdir(station_path) if f.endswith('.zarr') and os.path.isdir(os.path.join(station_path, f))]
                        if zarr_dirs:
                            # Use the first zarr directory found
                            zarr_path = os.path.join(station_path, zarr_dirs[0])
                            try:
                                ds = xr.open_zarr(zarr_path)
                                # Prefer transformed data if available
                                if station_dir not in processed_data or 'coordinate_system' in ds.attrs:
                                    processed_data[station_dir] = ds
                                    logger.info(f"  Found {'transformed' if 'coordinate_system' in ds.attrs else 'processed'} data for station {station_dir}")
                            except Exception as e:
                                logger.error(f"  Failed to open zarr: {e}")
        
        if not processed_data:
            logger.error("No processed data found to visualize")
            return
        
        # Find stations with coordinate transformation
        transformed_stations = []
        metadata_only_stations = []
        
        for station, ds in processed_data.items():
            if 'coordinate_system' in ds.attrs and ds.attrs['coordinate_system'] == 'cartesian':
                transformed_stations.append(station)
            else:
                metadata_only_stations.append(station)
        
        logger.info(f"Stations with coordinate transformation: {transformed_stations}")
        logger.info(f"Stations with metadata only: {metadata_only_stations}")
        
        # Create visualizations
        fig = plt.figure(figsize=(24, 14))
        
        # If we have transformed data, show polar to cartesian transformation
        if transformed_stations:
            station = transformed_stations[0]
            ds = processed_data[station]
            
            # Plot 1: Polar representation - show cartesian data in polar view
            ax1 = plt.subplot(2, 4, 1, projection='polar')
            
            # Use actual DBZH data to show in polar coordinates
            if 'DBZH' in ds.data_vars:
                dbzh = ds.DBZH.values
                
                # Get coordinate ranges (in degrees lat/lon)
                x = ds.x.values  # longitude
                y = ds.y.values  # latitude
                
                # Get radar center location
                radar_lon = float(ds.attrs.get('longitude', x.mean()))
                radar_lat = float(ds.attrs.get('latitude', y.mean()))
                
                # Create polar grid for visualization
                theta = np.linspace(0, 2*np.pi, 180)  # Azimuth angles
                # Maximum range in degrees (approximately 2 degrees covers ~220km)
                max_range_deg = 2.0
                r = np.linspace(0, max_range_deg, 100)  # Range in degrees
                
                # Create meshgrid for polar display
                theta_grid, r_grid = np.meshgrid(theta, r)
                
                # Convert polar grid to lat/lon coordinates
                # r is distance in degrees, theta is azimuth from north
                lat_polar = radar_lat + r_grid * np.cos(theta_grid)
                lon_polar = radar_lon + r_grid * np.sin(theta_grid)
                
                # Interpolate cartesian data to polar grid points
                from scipy.interpolate import RegularGridInterpolator
                
                # Create interpolator for the cartesian data
                # Note: RegularGridInterpolator expects (y, x) order for 2D data
                interpolator = RegularGridInterpolator(
                    (y, x), 
                    dbzh,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                
                # Sample at polar grid points (lat, lon order)
                points = np.column_stack([lat_polar.ravel(), lon_polar.ravel()])
                polar_data = interpolator(points).reshape(r_grid.shape)
                
                # Use a radar-like colormap
                try:
                    import pyart
                    cmap = pyart.graph.cm.NWSRef
                except ImportError:
                    # Create a custom radar colormap similar to NWS
                    colors_list = ['#00ffff', '#0080ff', '#0000ff', '#00ff00', '#00c000', '#008000',
                                  '#ffff00', '#ffc000', '#ff8000', '#ff0000', '#c00000', '#800000', 
                                  '#ff00ff', '#8000ff']
                    n_bins = 100
                    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('radar', colors_list, N=n_bins)
                
                # Mask invalid data
                polar_data_masked = np.ma.masked_invalid(polar_data)
                
                c = ax1.pcolormesh(theta_grid, r_grid, polar_data_masked, 
                                  cmap=cmap, vmin=-35, vmax=60, shading='auto')
                ax1.set_title(f'Polar View (from Cartesian)\n{station}', fontsize=12, fontweight='bold', pad=20)
                ax1.set_theta_zero_location('N')
                ax1.set_theta_direction(-1)
                ax1.set_ylim(0, r.max())
                
                # Add range rings
                for ring_r in np.linspace(0, r.max(), 5)[1:]:
                    ax1.plot(theta, [ring_r]*len(theta), 'k-', alpha=0.2, linewidth=0.5)
                
                # Add azimuth lines
                for angle in np.arange(0, 360, 30):
                    ax1.plot([np.radians(angle), np.radians(angle)], [0, r.max()], 'k-', alpha=0.2, linewidth=0.5)
                
                plt.colorbar(c, ax=ax1, label='dBZ', fraction=0.046, pad=0.04)
            else:
                ax1.text(0.5, 0.5, 'No DBZH data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f'Polar Coordinates\n{station}', fontsize=12, fontweight='bold')
            
            # Plot 2: Cartesian Grid with data overlay
            ax2 = plt.subplot(2, 4, 2)
            if 'x' in ds.coords and 'y' in ds.coords and 'DBZH' in ds.data_vars:
                x = ds.x.values
                y = ds.y.values
                dbzh = ds.DBZH.values
                
                # Plot the actual data with transparency to show grid
                if dbzh.ndim == 2:
                    dbzh_masked = np.ma.masked_invalid(dbzh)
                    
                    # Use same colormap
                    try:
                        import pyart
                        cmap = pyart.graph.cm.NWSRef
                    except ImportError:
                        colors_list = ['#00ffff', '#0080ff', '#0000ff', '#00ff00', '#00c000', '#008000',
                                      '#ffff00', '#ffc000', '#ff8000', '#ff0000', '#c00000', '#800000', 
                                      '#ff00ff', '#8000ff']
                        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('radar', colors_list, N=100)
                    
                    # Show data with some transparency
                    im = ax2.imshow(dbzh_masked, aspect='auto', cmap=cmap,
                                   extent=[x.min(), x.max(), y.min(), y.max()],
                                   origin='lower', vmin=-10, vmax=60, alpha=0.7)
                
                # Overlay grid lines
                for i in range(0, len(x), 20):
                    ax2.axvline(x[i], color='black', alpha=0.2, linewidth=0.5)
                for i in range(0, len(y), 20):
                    ax2.axhline(y[i], color='black', alpha=0.2, linewidth=0.5)
                
                ax2.set_title(f'Cartesian Grid\n{station}', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Longitude (°)')
                ax2.set_ylabel('Latitude (°)')
                ax2.set_xlim(x.min(), x.max())
                ax2.set_ylim(y.min(), y.max())
                
                # Add station marker at center
                ax2.plot(0, 0, 'r*', markersize=15, label='Radar Location')
                ax2.legend(loc='upper right')
            
            # Plot 3: Transformed DBZH Reflectivity in Cartesian
            ax3 = plt.subplot(2, 4, 3)
            if 'DBZH' in ds.data_vars and 'x' in ds.coords and 'y' in ds.coords:
                dbzh = ds.DBZH.values
                if dbzh.ndim == 2:
                    # Handle NaN values for better visualization
                    dbzh_masked = np.ma.masked_invalid(dbzh)
                    # Use same colormap as polar plot
                    try:
                        import pyart
                        cmap = pyart.graph.cm.NWSRef
                    except ImportError:
                        colors_list = ['#00ffff', '#0080ff', '#0000ff', '#00ff00', '#00c000', '#008000',
                                      '#ffff00', '#ffc000', '#ff8000', '#ff0000', '#c00000', '#800000', 
                                      '#ff00ff', '#8000ff']
                        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('radar', colors_list, N=100)
                    
                    im = ax3.imshow(dbzh_masked, aspect='auto', cmap=cmap, 
                                   extent=[ds.x.min().item(), ds.x.max().item(), 
                                          ds.y.min().item(), ds.y.max().item()],
                                   origin='lower', vmin=-10, vmax=60)
                    ax3.set_title(f'Cartesian DBZH\n{station}', fontsize=12, fontweight='bold')
                    ax3.set_xlabel('Longitude (°)')
                    ax3.set_ylabel('Latitude (°)')
                    plt.colorbar(im, ax=ax3, label='dBZ')
                    
                    # Add radar location
                    ax3.plot(0, 0, 'r*', markersize=10)
            else:
                ax3.text(0.5, 0.5, f'No DBZH data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title(f'No Reflectivity Data\n{station}', fontsize=12)
        
        else:
            # No transformed data, adjust layout
            ax1 = plt.subplot(2, 4, 1)
            ax1.axis('off')
            ax1.text(0.5, 0.5, 'No polar data available\n(Metadata-only processing)', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Polar Coordinates', fontsize=12, fontweight='bold')
            
            ax2 = plt.subplot(2, 4, 2)
            ax2.axis('off')
            ax2.text(0.5, 0.5, 'No cartesian grid\n(Metadata-only processing)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Cartesian Grid', fontsize=12, fontweight='bold')
            
            ax3 = plt.subplot(2, 4, 3)
            ax3.axis('off')
            ax3.text(0.5, 0.5, 'No reflectivity data\n(Metadata-only processing)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Transformed Data', fontsize=12, fontweight='bold')
        
        # Plot 4: Processing summary
        ax4 = plt.subplot(2, 4, 4)
        
        # Count stations and data types
        station_types = {
            'Transformed': len(transformed_stations),
            'Metadata Only': len(metadata_only_stations),
            'Total Processed': len(processed_data),
            'Original Data': len(original_data)
        }
        
        bars = ax4.bar(range(len(station_types)), list(station_types.values()), alpha=0.7, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax4.set_xticks(range(len(station_types)))
        ax4.set_xticklabels(list(station_types.keys()), rotation=45, ha='right')
        ax4.set_ylabel('Count')
        ax4.set_title('Processing Summary', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Plot 5: Data comparison table
        ax5 = plt.subplot(2, 4, 5)
        ax5.axis('off')
        
        # Create comparison table
        table_data = []
        table_data.append(['Station', 'Original', 'Processed', 'Transformed'])
        
        # Show up to 5 stations
        for station in list(processed_data.keys())[:5]:
            orig = '✓' if station in original_data else '✗'
            proc = '✓'  # Always true since we're iterating processed data
            trans = '✓' if station in transformed_stations else '✗'
            table_data.append([station, orig, proc, trans])
        
        table = ax5.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax5.set_title('Data Availability', fontsize=12, fontweight='bold', pad=20)
        
        # Plot 6: Metadata comparison
        ax6 = plt.subplot(2, 4, 6)
        ax6.axis('off')
        
        # Show metadata for first available station
        if processed_data:
            station = list(processed_data.keys())[0]
            ds = processed_data[station]
            
            metadata_text = f"Station: {station}\n"
            metadata_text += f"Variables: {len(ds.data_vars)}\n"
            metadata_text += f"Dimensions: {dict(ds.sizes)}\n"
            
            if 'processed' in ds.attrs:
                metadata_text += f"Processed: {ds.attrs['processed']}\n"
            if 'coordinate_system' in ds.attrs:
                metadata_text += f"Coordinate System: {ds.attrs['coordinate_system']}\n"
            if 'transformed' in ds.attrs:
                metadata_text += f"Transformed: {ds.attrs['transformed']}\n"
            if 'processing_timestamp' in ds.attrs:
                metadata_text += f"Processing Time: {ds.attrs['processing_timestamp'][:19]}\n"
            
            ax6.text(0.1, 0.9, metadata_text, fontsize=9, verticalalignment='top',
                    family='monospace', transform=ax6.transAxes)
            ax6.set_title(f'Sample Metadata ({station})', fontsize=12, fontweight='bold')
        
        # Plot 7: Variables comparison
        ax7 = plt.subplot(2, 4, 7)
        
        # Count variables across all processed datasets
        all_vars = set()
        var_counts = {}
        
        for station, ds in processed_data.items():
            for var in ds.data_vars:
                all_vars.add(var)
                if var not in var_counts:
                    var_counts[var] = 0
                var_counts[var] += 1
        
        if var_counts:
            sorted_vars = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            vars_names = [v[0] for v in sorted_vars]
            vars_counts = [v[1] for v in sorted_vars]
            
            bars = ax7.barh(range(len(vars_names)), vars_counts, alpha=0.7, color='#2E86AB')
            ax7.set_yticks(range(len(vars_names)))
            ax7.set_yticklabels(vars_names, fontsize=8)
            ax7.set_xlabel('Number of Stations')
            ax7.set_title('Variable Availability', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax7.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)}', ha='left', va='center', fontsize=8)
        
        # Plot 8: Coordinate system info
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        
        coord_info = "Coordinate Transformation:\n\n"
        coord_info += "Polar → Cartesian\n"
        coord_info += "━━━━━━━━━━━━━━━━\n\n"
        
        if transformed_stations:
            coord_info += f"✓ {len(transformed_stations)} stations transformed\n"
            coord_info += "• Grid: 100x100 points\n"
            coord_info += "• Range: ±230 km\n"
            coord_info += "• Resolution: ~4.6 km\n"
            coord_info += "• Projection: Cartesian\n"
        else:
            coord_info += "✗ No transformations\n"
            coord_info += "  (Metadata-only mode)\n"
        
        ax8.text(0.1, 0.9, coord_info, fontsize=10, verticalalignment='top',
                family='monospace', transform=ax8.transAxes)
        ax8.set_title('Transformation Info', fontsize=12, fontweight='bold')
        
        plt.suptitle('NEXRAD Processing Pipeline: Polar to Cartesian Transformation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(save_path, 'nexrad_comparison.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")
        plt.close()
        
        logger.info("Visualization complete!")
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to create visualizations."""
    
    logger.info("Creating visualizations of NEXRAD processing results...")
    
    # Paths relative to src/ directory
    data_path = '../data'
    output_path = '../out'
    transformed_path = '../out_with_transform'  # Path with actual transformed data
    save_path = '../visualizations'
    
    create_comparison_plots(data_path, output_path, save_path, transformed_path)


if __name__ == '__main__':
    main()