#!/usr/bin/env python3
"""
Script to download a random sample of NEXRAD Level II data from AWS S3
and save it as Zarr format using xarray and xradar libraries.
Downloads 3-5 files (~5-10MB total) for testing purposes.
"""

import os
import random
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import xarray as xr
import xradar
import zarr
from datetime import datetime, timedelta
import tempfile
import shutil


def get_random_nexrad_files(bucket_name='unidata-nexrad-level2', num_files=4):
    """Get random NEXRAD files from the AWS bucket."""
    
    # Create S3 client without credentials (public bucket)
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Get a random recent date (last 30 days to ensure data availability)
    end_date = datetime.now() - timedelta(days=1)  # Yesterday to ensure data exists
    start_date = end_date - timedelta(days=30)
    
    attempts = 0
    max_attempts = 10
    selected_files = []
    
    while len(selected_files) < num_files and attempts < max_attempts:
        attempts += 1
        
        # Generate random date
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        # Format date for S3 path
        date_path = random_date.strftime('%Y/%m/%d')
        
        try:
            # List radar sites for this date
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=f'{date_path}/',
                Delimiter='/'
            )
            
            if 'CommonPrefixes' not in response:
                continue
                
            # Get random radar site
            sites = [prefix['Prefix'].split('/')[-2] for prefix in response['CommonPrefixes']]
            if not sites:
                continue
                
            random_site = random.choice(sites)
            
            # List files for this site/date
            site_response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=f'{date_path}/{random_site}/'
            )
            
            if 'Contents' not in site_response:
                continue
                
            # Filter files and get a random one
            files = [obj for obj in site_response['Contents'] 
                    if obj['Key'].endswith('_V06') and obj['Size'] > 500000]  # At least 500KB
            
            if files:
                random_file = random.choice(files)
                file_info = {
                    'key': random_file['Key'],
                    'size': random_file['Size'],
                    'site': random_site,
                    'date': random_date.strftime('%Y-%m-%d')
                }
                selected_files.append(file_info)
                print(f"Selected: {file_info['key']} ({file_info['size']:,} bytes)")
        
        except Exception as e:
            print(f"Error accessing {date_path}: {e}")
            continue
    
    return selected_files


def download_and_convert_to_zarr(files, zarr_root_path='../data'):
    """Download NEXRAD files and create a unified Zarr store with radar sites as child groups."""
    
    # Remove existing zarr store if it exists and create fresh
    if os.path.exists(zarr_root_path):
        shutil.rmtree(zarr_root_path)
    
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'unidata-nexrad-level2'
    
    processed_sites = []
    total_original_size = 0
    
    # Create temporary directory for downloads  
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create the root Zarr store
        import zarr
        root_store = zarr.open_group(zarr_root_path, mode='w')
        
        # Add global metadata to root store
        root_store.attrs.update({
            'title': 'NEXRAD Level II Sample Data Collection',
            'description': 'Multi-site NEXRAD Level II radar data samples',
            'source': 'AWS NEXRAD Open Data Program',
            'creation_time': datetime.now().isoformat(),
            'format_version': '1.0',
            'sites_included': [f['site'] for f in files]
        })
        
        for i, file_info in enumerate(files):
            try:
                filename = os.path.basename(file_info['key'])
                temp_path = os.path.join(temp_dir, filename)
                site = file_info['site']
                
                print(f"Downloading {filename} for site {site}...")
                
                # Download file
                s3_client.download_file(bucket_name, file_info['key'], temp_path)
                actual_size = os.path.getsize(temp_path)
                total_original_size += actual_size
                
                print(f"✓ Downloaded {filename} ({actual_size:,} bytes)")
                print(f"Processing data for site {site}...")
                
                # Load with xradar
                import xradar as xd
                dt = xd.io.open_nexradlevel2_datatree(temp_path)
                print(f"  Datatree loaded with {len(dt.children)} sweeps")
                
                # Create site group in root Zarr store
                site_group = root_store.create_group(site, overwrite=True)
                
                # Get the root dataset (contains shared schema variables)
                root_ds = dt.ds
                if root_ds is not None:
                    print(f"  Root dataset variables: {list(root_ds.data_vars.keys())}")
                    
                    # Save root dataset variables to the site group
                    root_ds.to_zarr(zarr_root_path, group=site, mode='a')
                
                # Process each sweep and add to site group
                sweep_count = 0
                for sweep_name, sweep_node in dt.children.items():
                    if sweep_node.ds is not None:
                        sweep_ds = sweep_node.ds
                        
                        # Create sweep subgroup
                        sweep_group_path = f"{site}/sweep_{sweep_count:02d}"
                        
                        # Add sweep metadata
                        sweep_attrs = {
                            'sweep_name': sweep_name,
                            'sweep_number': sweep_count,
                            'elevation_angle': float(sweep_ds.attrs.get('fixed_angle', 0)),
                        }
                        
                        # Save sweep data to subgroup
                        sweep_ds.to_zarr(zarr_root_path, group=sweep_group_path, mode='a')
                        
                        # Update subgroup attributes
                        sweep_zarr_group = zarr.open_group(zarr_root_path)[site][f'sweep_{sweep_count:02d}']
                        sweep_zarr_group.attrs.update(sweep_attrs)
                        
                        sweep_count += 1
                
                # Update site group metadata
                site_zarr_group = zarr.open_group(zarr_root_path)[site]
                site_zarr_group.attrs.update({
                    'site_id': site,
                    'original_filename': filename,
                    'date': file_info['date'],
                    'original_size_bytes': actual_size,
                    'num_sweeps': sweep_count,
                    'processing_time': datetime.now().isoformat()
                })
                
                print(f"✓ Processed {site} with {sweep_count} sweeps")
                
                # Close datatree
                dt.close()
                
                processed_sites.append({
                    'site': site,
                    'original_size': actual_size,
                    'date': file_info['date'],
                    'original_filename': filename,
                    'sweeps': sweep_count
                })
                
            except Exception as e:
                print(f"✗ Error processing {file_info['key']}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate total Zarr size
    total_zarr_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(zarr_root_path)
        for filename in filenames
    ) if os.path.exists(zarr_root_path) else 0
    
    return processed_sites, total_original_size, total_zarr_size


def verify_zarr_structure(zarr_root_path, processed_sites):
    """Verify that the unified Zarr store can be read correctly."""
    
    print(f"\nVerifying unified Zarr structure...")
    
    try:
        # Open root Zarr store
        import zarr
        root_store = zarr.open_group(zarr_root_path, mode='r')
        
        print(f"✓ Root Zarr store opened successfully")
        print(f"  - Root attributes: {dict(root_store.attrs)}")
        print(f"  - Site groups: {list(root_store.group_keys())}")
        
        # Verify each site
        for site_info in processed_sites:
            site = site_info['site']
            try:
                site_group = root_store[site]
                print(f"✓ Site {site}:")
                print(f"  - Site attributes: {dict(site_group.attrs)}")
                print(f"  - Subgroups: {list(site_group.group_keys())}")
                
                # Try to read site data with xarray
                site_ds = xr.open_zarr(zarr_root_path, group=site)
                print(f"  - Root variables: {list(site_ds.data_vars.keys())[:5]}...")
                print(f"  - Dimensions: {dict(site_ds.sizes)}")
                site_ds.close()
                
                # Check a few sweeps
                sweep_groups = [k for k in site_group.group_keys() if k.startswith('sweep_')]
                if sweep_groups:
                    sample_sweep = sweep_groups[0]
                    sweep_ds = xr.open_zarr(zarr_root_path, group=f"{site}/{sample_sweep}")
                    print(f"  - Sample sweep ({sample_sweep}) vars: {list(sweep_ds.data_vars.keys())[:3]}...")
                    print(f"  - Sample sweep dims: {dict(sweep_ds.sizes)}")
                    sweep_ds.close()
                
            except Exception as e:
                print(f"✗ Error verifying site {site}: {e}")
    
    except Exception as e:
        print(f"✗ Error opening root Zarr store: {e}")


def main():
    """Main function to download NEXRAD data and create unified Zarr store."""
    
    print("NEXRAD Level II Data Sample Downloader (Unified Zarr Format)")
    print("=" * 70)
    
    # Get random files
    print("Finding random NEXRAD files...")
    files = get_random_nexrad_files(num_files=4)
    
    if not files:
        print("Could not find any suitable files. Please try again.")
        return
    
    print(f"\nFound {len(files)} files:")
    total_estimated_size = sum(f['size'] for f in files)
    print(f"Total estimated size: {total_estimated_size:,} bytes ({total_estimated_size/1024/1024:.1f} MB)")
    for f in files:
        print(f"  • {f['site']} - {f['date']} - {f['size']:,} bytes")
    
    # Download and convert to unified Zarr
    zarr_root_path = '../data'
    print(f"\nCreating unified Zarr store at {zarr_root_path}...")
    processed_sites, actual_total_size, total_zarr_size = download_and_convert_to_zarr(files, zarr_root_path)
    
    print(f"\nConversion Summary:")
    print(f"  - Sites processed: {len(processed_sites)}")
    print(f"  - Total original size: {actual_total_size:,} bytes ({actual_total_size/1024/1024:.1f} MB)")
    print(f"  - Total Zarr size: {total_zarr_size:,} bytes ({total_zarr_size/1024/1024:.1f} MB)")
    print(f"  - Zarr root store: {zarr_root_path}")
    
    if processed_sites:
        print(f"\nProcessed radar sites:")
        for site_info in processed_sites:
            print(f"    • {site_info['site']} - {site_info['date']} - {site_info['sweeps']} sweeps")
    
    # Verify unified Zarr structure
    if processed_sites:
        verify_zarr_structure(zarr_root_path, processed_sites)
    
    print(f"\n✓ Unified Zarr store creation complete!")
    print(f"✓ Access the data with:")
    print(f"  - Root store: xr.open_zarr('{zarr_root_path}')")
    print(f"  - Specific site: xr.open_zarr('{zarr_root_path}', group='SITE_NAME')")
    print(f"  - Specific sweep: xr.open_zarr('{zarr_root_path}', group='SITE_NAME/sweep_00')")


if __name__ == "__main__":
    main()
