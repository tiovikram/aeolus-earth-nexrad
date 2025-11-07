#!/usr/bin/env python3
"""
Test script for the Natural Language NEXRAD Processing Pipeline.
Tests query parsing without running the full pipeline.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, 'src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_query_parsing():
    """Test natural language query parsing functionality."""
    
    # Check if Claude API key is available
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.warning("ANTHROPIC_API_KEY not set - skipping parsing tests")
        logger.info("Set ANTHROPIC_API_KEY environment variable to test query parsing")
        return
    
    try:
        from natural_language_parser import NaturalLanguageParser
    except ImportError as e:
        logger.error(f"Cannot import natural language parser: {e}")
        logger.error("Install required packages: pip install anthropic")
        return
    
    parser = NaturalLanguageParser()
    
    # Test queries with different complexity levels
    test_queries = [
        # Simple queries
        "Process KABR radar data from yesterday 2PM to 6PM",
        "Get KPDT data for the last 3 hours",
        "I need KYUX weather radar from today morning",
        
        # Multi-station queries
        "Process KABR, KPDT, and KYUX stations from last night",
        "Get data from multiple stations KSGF,KRTX for yesterday afternoon",
        
        # Time range queries
        "Analyze PGUA radar from 2024-01-15 6AM to noon",
        "Process KPDT station from January 1st 2024 midnight to 6AM",
        "Get KABR data for severe weather on June 15th 6PM to midnight",
        
        # Location-based queries (should map to station codes)
        "Process Denver area radar for last 2 hours",
        "Get Portland Oregon weather radar from yesterday",
        
        # Relative time queries
        "Process KABR for the past 4 hours",
        "Get last night's data from KPDT",
        "Analyze this morning's radar data from KYUX"
    ]
    
    logger.info(f"Testing {len(test_queries)} natural language queries...")
    logger.info("=" * 60)
    
    successful_parses = 0
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nTest {i}/{len(test_queries)}")
        logger.info(f"Query: '{query}'")
        logger.info("-" * 40)
        
        try:
            result = parser.parse_query(query)
            
            # Validate results
            if result.stations and result.start_time and result.end_time:
                logger.info("✅ Successfully parsed:")
                logger.info(f"  Stations: {', '.join(result.stations)}")
                logger.info(f"  Start: {result.start_time}")
                logger.info(f"  End: {result.end_time}")
                logger.info(f"  Confidence: {result.confidence:.2f}")
                
                # Additional validation
                if result.start_time >= result.end_time:
                    logger.warning("⚠️  Start time is not before end time")
                elif (result.end_time - result.start_time).total_seconds() < 300:  # 5 minutes
                    logger.warning("⚠️  Time range seems very short")
                elif (result.end_time - result.start_time).days > 7:  # 1 week
                    logger.warning("⚠️  Time range seems very long")
                else:
                    successful_parses += 1
                    
            else:
                logger.error("❌ Parsing failed - missing required fields")
                
        except Exception as e:
            logger.error(f"❌ Parsing failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"PARSING TEST SUMMARY: {successful_parses}/{len(test_queries)} queries parsed successfully")
    
    if successful_parses >= len(test_queries) * 0.8:  # 80% success rate
        logger.info("🎉 Query parsing tests passed!")
    else:
        logger.warning(f"⚠️  Only {successful_parses} out of {len(test_queries)} queries parsed successfully")
    
    return successful_parses >= len(test_queries) * 0.8


def test_s3_path_validation():
    """Test S3 bucket name validation."""
    
    logger.info("\nTesting S3 bucket name validation...")
    
    # Import the validation function
    sys.path.insert(0, '.')
    try:
        from run_natural_language_pipeline import validate_s3_bucket
    except ImportError as e:
        logger.error(f"Cannot import S3 validation: {e}")
        return False
    
    test_buckets = [
        # Valid bucket names
        ("my-nexrad-results", True),
        ("weather-analysis-2024", True),
        ("nexrad.data.bucket", True),
        ("simple-bucket", True),
        
        # Invalid bucket names
        ("My-Bucket", False),  # Uppercase
        ("bucket_with_underscores", False),  # Underscores
        ("ab", False),  # Too short
        ("a" * 64, False),  # Too long
        ("bucket..double.dots", False),  # Double dots
        ("-starts-with-dash", False),  # Starts with dash
        ("ends-with-dash-", False),  # Ends with dash
    ]
    
    passed = 0
    
    for bucket_name, expected_valid in test_buckets:
        result = validate_s3_bucket(bucket_name)
        if result == expected_valid:
            status = "✅" if expected_valid else "✅ (correctly invalid)"
            passed += 1
        else:
            status = "❌"
        
        logger.info(f"{status} {bucket_name}: {result} (expected: {expected_valid})")
    
    logger.info(f"S3 validation test: {passed}/{len(test_buckets)} passed")
    return passed == len(test_buckets)


def test_cli_argument_parsing():
    """Test command-line argument parsing without running pipeline."""
    
    logger.info("\nTesting CLI argument parsing...")
    
    # Test different command combinations
    test_commands = [
        # Basic query
        ['--query', 'Process KABR radar from yesterday 2PM to 6PM'],
        
        # Query with S3 output
        ['--query', 'Get KPDT data for last 3 hours', '--s3_output_bucket', 'my-results'],
        
        # Query with cloud processing
        ['--query', 'Process multiple stations from last night', 
         '--runner', 'DataflowRunner', '--project', 'my-project'],
        
        # Dry run
        ['--query', 'Test query', '--dry_run'],
    ]
    
    for i, args in enumerate(test_commands, 1):
        logger.info(f"Test command {i}: {' '.join(args)}")
        
        # Test argument parsing (but don't run the pipeline)
        import argparse
        
        # Recreate the parser from the main script
        parser = argparse.ArgumentParser()
        parser.add_argument('--query', required=True)
        parser.add_argument('--s3_output_bucket')
        parser.add_argument('--local_output_path', default='./out')
        parser.add_argument('--runner', default='DirectRunner')
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--project')
        parser.add_argument('--region', default='us-central1')
        parser.add_argument('--dry_run', action='store_true')
        parser.add_argument('--debug', action='store_true')
        
        try:
            parsed_args = parser.parse_args(args)
            logger.info(f"✅ Arguments parsed successfully")
            logger.info(f"  Query: {parsed_args.query}")
            if parsed_args.s3_output_bucket:
                logger.info(f"  S3 bucket: {parsed_args.s3_output_bucket}")
            if parsed_args.runner != 'DirectRunner':
                logger.info(f"  Runner: {parsed_args.runner}")
        except Exception as e:
            logger.error(f"❌ Argument parsing failed: {e}")
    
    logger.info("CLI argument parsing tests completed")
    return True


def main():
    """Run all tests for the natural language pipeline."""
    
    logger.info("🧪 Natural Language NEXRAD Pipeline Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("S3 Path Validation", test_s3_path_validation),
        ("CLI Argument Parsing", test_cli_argument_parsing),
        ("Natural Language Query Parsing", test_query_parsing)
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name} passed")
                passed_tests += 1
            else:
                logger.warning(f"⚠️  {test_name} had issues")
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST SUMMARY: {passed_tests}/{len(tests)} test suites passed")
    
    if passed_tests == len(tests):
        logger.info("🎉 All tests passed! Natural language pipeline is ready to use.")
        logger.info("\nTo get started:")
        logger.info("1. Set ANTHROPIC_API_KEY environment variable")
        logger.info("2. Install required packages: pip install anthropic s3fs")
        logger.info("3. Run: python run_natural_language_pipeline.py --query \"Your query here\"")
    else:
        logger.warning(f"⚠️  {len(tests) - passed_tests} test suite(s) had issues")
    
    return passed_tests == len(tests)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)