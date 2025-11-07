#!/usr/bin/env python3
"""
Natural Language NEXRAD Processing Pipeline Runner.
Accepts plain text queries like "Process KABR radar data from yesterday 2PM to 6PM"
and automatically parses them into structured pipeline arguments.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from apache_beam.options.pipeline_options import PipelineOptions

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # dotenv not available, skip

# Add proper path handling for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Also add parent directory in case we're running from project root
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from natural_language_parser import NaturalLanguageParser, ParsedQuery
    from enhanced_nexrad_pipeline import NexradPipelineOptions, process_nexrad_from_aws
    PARSER_AVAILABLE = True
except ImportError as e:
    PARSER_AVAILABLE = False
    logging.error(f"Import error: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_s3_bucket(bucket_name: str) -> bool:
    """Validate S3 bucket name format."""
    import re
    
    # Basic S3 bucket name validation
    if not bucket_name:
        return False
    
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


def setup_s3_output(bucket_name: str) -> str:
    """Setup S3 output path and validate access."""
    
    if not validate_s3_bucket(bucket_name):
        raise ValueError(f"Invalid S3 bucket name: {bucket_name}")
    
    s3_path = f"s3://{bucket_name}/nexrad-results"
    
    # Test S3 access (optional - could be expensive)
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client('s3')
        
        # Try to list objects to verify access
        try:
            s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            logger.info(f"✓ S3 bucket '{bucket_name}' is accessible")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                logger.warning(f"S3 bucket '{bucket_name}' does not exist - will be created during processing")
            else:
                logger.warning(f"Cannot verify S3 bucket access: {e}")
        
    except ImportError:
        logger.warning("boto3 not available - cannot validate S3 access")
    
    return s3_path


def create_pipeline_from_parsed_query(parsed_query: ParsedQuery, 
                                     output_location: str,
                                     runner_args: dict) -> dict:
    """Create pipeline configuration from parsed natural language query."""
    
    # Convert ParsedQuery to pipeline configuration
    pipeline_config = {
        'stations': ','.join(parsed_query.stations),
        'start_time': parsed_query.start_time.strftime('%Y-%m-%d-%H:%M'),
        'end_time': parsed_query.end_time.strftime('%Y-%m-%d-%H:%M'),
        'output_path': output_location
    }
    
    # Add default processing parameters
    pipeline_config.update({
        'max_files_per_station': 10,
        'grid_size': 100,
        'include_sweeps': '0,1,2'
    })
    
    # Add any additional parameters from query parsing
    if parsed_query.additional_params:
        for key, value in parsed_query.additional_params.items():
            if key not in ['stations', 'start_time', 'end_time', 'confidence', 'reasoning']:
                pipeline_config[key] = value
    
    # Add runner configuration
    pipeline_config.update(runner_args)
    
    return pipeline_config


def main():
    """Main function with natural language query support."""
    
    parser = argparse.ArgumentParser(
        description='Natural Language NEXRAD Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic natural language query with local output
    python run_natural_language_pipeline.py \\
        --query "Process KABR radar data from yesterday 2PM to 6PM"
    
    # Natural language query with S3 output
    python run_natural_language_pipeline.py \\
        --query "Get KPDT and KYUX data for the last 3 hours" \\
        --s3_output_bucket my-nexrad-results
    
    # More complex query
    python run_natural_language_pipeline.py \\
        --query "I need weather radar from KSGF station covering today 6AM to noon" \\
        --s3_output_bucket weather-analysis-bucket \\
        --runner DataflowRunner \\
        --project my-gcp-project
    
    # Query examples:
    - "Process KABR radar data from yesterday 2PM to 6PM"
    - "Get KPDT and KYUX data for the last 3 hours"  
    - "Analyze multiple stations KABR,KPDT,KYUX from last night"
    - "I need PGUA weather radar from 2024-01-15 morning"
    - "Process Denver area radar for severe weather on June 15th 6PM to midnight"
        """
    )
    
    # Natural language query
    parser.add_argument('--query', required=True,
                       help='Natural language description of radar data processing request')
    
    # Output configuration
    parser.add_argument('--s3_output_bucket', 
                       help='S3 bucket name for output storage (e.g., my-nexrad-results)')
    parser.add_argument('--local_output_path', default='./out',
                       help='Local output directory if no S3 bucket specified')
    
    # Apache Beam options
    parser.add_argument('--runner', default='DirectRunner',
                       help='Apache Beam runner (DirectRunner, DataflowRunner)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of workers')
    parser.add_argument('--project', help='GCP project (for DataflowRunner)')
    parser.add_argument('--region', default='us-central1',
                       help='GCP region (for DataflowRunner)')
    
    # Testing and debugging
    parser.add_argument('--dry_run', action='store_true',
                       help='Parse query and show configuration without running pipeline')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate dependencies
    if not PARSER_AVAILABLE:
        logger.error("Required dependencies not available.")
        logger.error("Install required packages: pip install anthropic apache-beam[gcp]")
        return 1
    
    # Validate Claude API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        logger.error("Get API key from: https://console.anthropic.com/")
        return 1
    
    logger.info("🧠 Natural Language NEXRAD Processing Pipeline")
    logger.info(f"Query: '{args.query}'")
    
    try:
        # Parse natural language query
        logger.info("Parsing natural language query with Claude API...")
        parser_instance = NaturalLanguageParser()
        parsed_query = parser_instance.parse_query(args.query)
        
        logger.info(f"✓ Parsed query successfully (confidence: {parsed_query.confidence:.2f})")
        logger.info(f"  Stations: {parsed_query.stations}")
        logger.info(f"  Time range: {parsed_query.start_time} to {parsed_query.end_time}")
        
        # Determine output location
        if args.s3_output_bucket:
            output_location = setup_s3_output(args.s3_output_bucket)
            logger.info(f"Output destination: S3 bucket '{args.s3_output_bucket}'")
        else:
            output_location = args.local_output_path
            logger.info(f"Output destination: Local directory '{output_location}'")
        
        # Create runner configuration
        runner_config = {
            'runner': args.runner,
            'num_workers': args.num_workers
        }
        
        if args.project:
            runner_config['project'] = args.project
            runner_config['region'] = args.region
        
        # Create pipeline configuration from parsed query
        pipeline_config = create_pipeline_from_parsed_query(
            parsed_query, output_location, runner_config
        )
        
        logger.info("Pipeline configuration:")
        for key, value in pipeline_config.items():
            logger.info(f"  {key}: {value}")
        
        if args.dry_run:
            logger.info("🏃 Dry run mode - configuration validated successfully")
            return 0
        
        # Build Apache Beam arguments
        beam_args = []
        for key, value in pipeline_config.items():
            if key in ['runner', 'project', 'region', 'num_workers']:
                if key == 'num_workers':
                    beam_args.append(f"--direct_num_workers={value}")
                else:
                    beam_args.append(f"--{key}={value}")
            else:
                beam_args.append(f"--{key}={value}")
        
        # Add additional Beam configuration
        beam_args.extend([
            "--direct_running_mode=multi_processing",
            "--save_main_session"
        ])
        
        # Add DataflowRunner specific options
        if args.runner == 'DataflowRunner':
            if not args.project:
                logger.error("--project is required for DataflowRunner")
                return 1
            
            job_name = f"nexrad-nlp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            beam_args.extend([
                f"--temp_location=gs://{args.project}/temp",
                f"--staging_location=gs://{args.project}/staging", 
                f"--job_name={job_name}"
            ])
        
        logger.info("🚀 Starting NEXRAD processing pipeline...")
        logger.info(f"Beam arguments: {beam_args}")
        
        # Create and run pipeline
        pipeline_options = PipelineOptions(beam_args)
        nexrad_options = pipeline_options.view_as(NexradPipelineOptions)
        
        # Run the enhanced pipeline
        process_nexrad_from_aws(nexrad_options)
        
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"Results saved to: {output_location}")
        
        if args.s3_output_bucket:
            logger.info(f"Check S3 bucket: https://s3.console.aws.amazon.com/s3/buckets/{args.s3_output_bucket}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())