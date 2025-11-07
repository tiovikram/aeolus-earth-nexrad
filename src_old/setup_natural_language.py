#!/usr/bin/env python3
"""
Setup script for Natural Language NEXRAD Processing Pipeline.
Installs required dependencies and validates the setup.
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_package(package_name):
    """Install a Python package using pip."""
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        logger.info(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        logger.error(f"❌ Failed to install {package_name}")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is available."""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def main():
    """Setup the natural language pipeline environment."""
    
    logger.info("🚀 Setting up Natural Language NEXRAD Processing Pipeline")
    logger.info("=" * 60)
    
    # Required packages
    packages = [
        ('python-dotenv', 'dotenv'),  # (pip_name, import_name)
        ('anthropic', 'anthropic'),
        ('s3fs', 's3fs'),
        ('apache-beam[gcp]', 'apache_beam'),
        ('xarray', 'xarray'),
        ('xarray-beam', 'xarray_beam'),
        ('zarr', 'zarr'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('boto3', 'boto3'),
        ('pandas', 'pandas'),
    ]
    
    # Check what's already installed
    logger.info("Checking existing packages...")
    installed = []
    missing = []
    
    for pip_name, import_name in packages:
        if check_package(import_name):
            logger.info(f"✅ {import_name} is already installed")
            installed.append(pip_name)
        else:
            logger.info(f"❌ {import_name} is missing")
            missing.append((pip_name, import_name))
    
    # Install missing packages
    if missing:
        logger.info(f"\n📦 Installing {len(missing)} missing packages...")
        failed_installations = []
        
        for pip_name, import_name in missing:
            if not install_package(pip_name):
                failed_installations.append(pip_name)
        
        if failed_installations:
            logger.error(f"❌ Failed to install: {', '.join(failed_installations)}")
            logger.error("Please install these packages manually:")
            for pkg in failed_installations:
                logger.error(f"  pip install {pkg}")
            return False
    else:
        logger.info("✅ All required packages are already installed")
    
    # Check .env file
    logger.info("\n🔧 Checking .env file...")
    env_file = '.env'
    
    if os.path.exists(env_file):
        logger.info(f"✅ Found {env_file}")
        
        # Check if API key is set
        with open(env_file, 'r') as f:
            content = f.read()
            
        if 'ANTHROPIC_API_KEY=your_api_key_here' in content:
            logger.warning("⚠️  ANTHROPIC_API_KEY still has placeholder value")
            logger.warning("Please update your .env file with your actual Claude API key")
            logger.info("Get your API key from: https://console.anthropic.com/")
        elif 'ANTHROPIC_API_KEY=' in content and len(content.split('ANTHROPIC_API_KEY=')[1].split('\n')[0].strip()) > 10:
            logger.info("✅ ANTHROPIC_API_KEY appears to be set in .env")
        else:
            logger.warning("⚠️  ANTHROPIC_API_KEY not found or appears empty in .env")
    else:
        logger.warning(f"⚠️  {env_file} not found")
        logger.info("A template .env file should have been created for you")
    
    # Test basic functionality
    logger.info("\n🧪 Testing basic functionality...")
    
    try:
        # Test imports
        sys.path.insert(0, 'src')
        from natural_language_parser import NaturalLanguageParser
        logger.info("✅ Natural language parser module loads correctly")
        
        # Test argument parsing
        from run_natural_language_pipeline import validate_s3_bucket
        if validate_s3_bucket('test-bucket-name'):
            logger.info("✅ S3 bucket validation works")
        else:
            logger.warning("⚠️  S3 bucket validation issue")
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 60)
    
    logger.info("✅ Required packages installed")
    logger.info("✅ Natural language parser module working")
    
    if os.path.exists(env_file):
        logger.info("✅ .env file exists")
    else:
        logger.warning("⚠️  .env file missing")
    
    logger.info("\n🎉 Setup complete!")
    logger.info("\nNext steps:")
    logger.info("1. Add your Claude API key to the .env file")
    logger.info("2. Test with: python test_natural_language_pipeline.py")
    logger.info("3. Try: python run_natural_language_pipeline.py --query 'Process KABR from yesterday' --dry_run")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)