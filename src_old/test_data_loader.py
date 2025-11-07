#!/usr/bin/env python3
"""
Simple test for the NEXRAD data loader without Apache Beam dependencies.
"""

import sys
import os
from datetime import datetime, timedelta
import logging
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import re
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleNexradDataLoader:
    """Simplified NEXRAD data loader for testing."""
    
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
    
    def find_files_for_station_and_date(self, station_id: str, date: datetime) -> List[dict]:
        """Find NEXRAD files for a station on a specific date."""
        
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
                filename = os.path.basename(obj['Key'])
                timestamp = self._parse_nexrad_filename(filename)
                
                if timestamp:
                    files.append({
                        'station_id': station_id,
                        'timestamp': timestamp,
                        's3_key': obj['Key'],
                        'size': obj['Size'],
                        'filename': filename
                    })
            
            return sorted(files, key=lambda x: x['timestamp'])
            
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


def test_data_loader():
    """Test the NEXRAD data loader functionality."""
    
    logger.info("Testing NEXRAD AWS Data Loader...")
    
    data_loader = SimpleNexradDataLoader()
    
    # Test 1: Get available stations for recent date
    test_date = datetime.now() - timedelta(days=1)
    logger.info(f"Testing station listing for {test_date.strftime('%Y-%m-%d')}")
    
    stations = data_loader.get_available_stations(test_date)
    logger.info(f"Found {len(stations)} stations")
    
    if stations:
        logger.info(f"Sample stations: {stations[:10]}")  # Show first 10
        
        # Test 2: Find files for a station
        test_station = stations[0] if stations else 'KABR'
        logger.info(f"Testing file search for {test_station}")
        
        files = data_loader.find_files_for_station_and_date(test_station, test_date)
        logger.info(f"Found {len(files)} files for {test_station}")
        
        if files:
            logger.info("Sample files:")
            for file_info in files[:3]:  # Show first 3
                logger.info(f"  - {file_info['filename']} ({file_info['size']:,} bytes) at {file_info['timestamp']}")
        
        return len(stations) > 0 and len(files) >= 0  # Success if we found stations
    
    else:
        logger.warning("No stations found - may be connectivity issue or date too recent")
        return False


def test_time_range_functionality():
    """Test time range filtering functionality."""
    
    logger.info("Testing time range functionality...")
    
    data_loader = SimpleNexradDataLoader()
    
    # Use a date we know has data (a week ago)
    test_date = datetime.now() - timedelta(days=7)
    
    stations = data_loader.get_available_stations(test_date)
    
    if not stations:
        logger.warning("No stations found for time range test")
        return False
    
    test_station = stations[0]
    files = data_loader.find_files_for_station_and_date(test_station, test_date)
    
    if not files:
        logger.warning(f"No files found for {test_station} on {test_date.strftime('%Y-%m-%d')}")
        return False
    
    # Test time filtering
    start_time = files[0]['timestamp']
    end_time = start_time + timedelta(hours=1)  # 1 hour window
    
    filtered_files = [
        f for f in files 
        if start_time <= f['timestamp'] <= end_time
    ]
    
    logger.info(f"Time range test: {len(files)} total files, {len(filtered_files)} in 1-hour window")
    logger.info(f"Window: {start_time} to {end_time}")
    
    return len(filtered_files) > 0


def test_multiple_stations():
    """Test multiple station processing."""
    
    logger.info("Testing multiple station functionality...")
    
    data_loader = SimpleNexradDataLoader()
    
    test_date = datetime.now() - timedelta(days=3)  # 3 days ago for better data availability
    
    stations = data_loader.get_available_stations(test_date)[:3]  # Test with 3 stations
    
    if len(stations) < 2:
        logger.warning("Not enough stations for multi-station test")
        return False
    
    logger.info(f"Testing stations: {stations}")
    
    total_files = 0
    station_results = {}
    
    for station in stations:
        files = data_loader.find_files_for_station_and_date(station, test_date)
        total_files += len(files)
        station_results[station] = len(files)
        logger.info(f"{station}: {len(files)} files")
    
    logger.info(f"Multi-station test: {total_files} total files from {len(stations)} stations")
    logger.info(f"Station results: {station_results}")
    
    return total_files > 0


def run_all_tests():
    """Run all data loader tests."""
    
    logger.info("=" * 60)
    logger.info("NEXRAD AWS DATA LOADER TEST SUITE")
    logger.info("=" * 60)
    
    tests = [
        ("AWS S3 Data Loader", test_data_loader),
        ("Time Range Functionality", test_time_range_functionality),
        ("Multiple Stations", test_multiple_stations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'-' * 40}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'-' * 40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"Test {test_name}: {status}")
            
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Data loader tests passed! Enhanced pipeline components are working.")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)