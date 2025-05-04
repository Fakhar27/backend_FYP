import os
import sys
import shutil
import subprocess
import logging
import asyncio
from typing import Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = "E:/fyp_backend/backend"
GENAI_DIR = os.path.join(PROJECT_DIR, "genAI")
SERVICES_DIR = os.path.join(GENAI_DIR, "services")
LANGCHAIN_SERVICE_PATH = os.path.join(SERVICES_DIR, "langchain_service.py")
BACKUP_PATH = os.path.join(SERVICES_DIR, "langchain_service.py.backup")
PATCHED_PATH = os.path.join(SERVICES_DIR, "langchain_service.py.patched")

def backup_original_file():
    """Create a backup of the original langchain_service.py file"""
    logger.info(f"Creating backup of {LANGCHAIN_SERVICE_PATH}")
    if os.path.exists(BACKUP_PATH):
        logger.info(f"Backup already exists at {BACKUP_PATH}")
    else:
        shutil.copy(LANGCHAIN_SERVICE_PATH, BACKUP_PATH)
        logger.info(f"Backup created at {BACKUP_PATH}")

def restore_original_file():
    """Restore the original langchain_service.py file from backup"""
    logger.info(f"Restoring original file from {BACKUP_PATH}")
    if os.path.exists(BACKUP_PATH):
        shutil.copy(BACKUP_PATH, LANGCHAIN_SERVICE_PATH)
        logger.info(f"Original file restored")
    else:
        logger.error(f"Backup not found at {BACKUP_PATH}")

def apply_patch():
    """Apply the patch to langchain_service.py"""
    logger.info("Applying patch to langchain_service.py")
    
    # Run the patch script
    patch_script = os.path.join(GENAI_DIR, "patch_langchain_service.py")
    result = subprocess.run([sys.executable, patch_script], 
                           cwd=GENAI_DIR,
                           capture_output=True,
                           text=True)
    
    logger.info(result.stdout)
    if result.stderr:
        logger.error(result.stderr)
    
    # Check if patch was created
    if os.path.exists(PATCHED_PATH):
        # Replace the original file with the patched version
        shutil.copy(PATCHED_PATH, LANGCHAIN_SERVICE_PATH)
        logger.info("Patch applied successfully")
        return True
    else:
        logger.error("Patch was not created")
        return False

def run_test():
    """Run the test script"""
    logger.info("Running the test script")
    
    # Run the test script
    test_script = os.path.join(SERVICES_DIR, "test_wan_video.py")
    result = subprocess.run([sys.executable, test_script], 
                           cwd=GENAI_DIR,
                           capture_output=True,
                           text=True)
    
    logger.info("Test output:")
    logger.info(result.stdout)
    if result.stderr:
        logger.error("Test errors:")
        logger.error(result.stderr)
    
    # Check if test passed
    if "Error in test_wan_video_generation" in result.stdout or result.returncode != 0:
        logger.error("Test failed")
        return False
    else:
        logger.info("Test passed")
        return True

def main():
    """Main function to run the tests"""
    logger.info("==== Starting WAN Video Test ====")
    
    # Create backup of original file
    backup_original_file()
    
    try:
        # Apply patch
        patch_success = apply_patch()
        if not patch_success:
            logger.error("Failed to apply patch, aborting test")
            return
        
        # Run test
        test_success = run_test()
        if test_success:
            logger.info("==== Test completed successfully! ====")
        else:
            logger.error("==== Test failed! ====")
    
    finally:
        # Always restore the original file when done
        restore_original_file()

if __name__ == "__main__":
    main()
