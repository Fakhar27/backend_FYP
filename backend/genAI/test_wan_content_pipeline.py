import asyncio
import json
import os
import logging
import sys
import base64
import traceback
import time
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import the service and request model
from services.langchain_service import StoryIterationChain, ContentRequest

class WanVideoGenerator:
    """Class to handle WAN video generation with pipeline-like processing"""
    
    def __init__(self):
        self.story_chain = None
        self.output_directory = os.path.join(os.path.dirname(__file__), "output")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
    async def initialize_service(self):
        """Initialize the StoryIterationChain service"""
        try:
            self.story_chain = StoryIterationChain()
            logger.info("Successfully initialized StoryIterationChain")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize StoryIterationChain: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    async def generate_video(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate video using WAN API with pipeline processing
        
        Args:
            request_data: Dictionary containing request parameters
            
        Returns:
            Result dictionary with success/failure and metadata
        """
        if not self.story_chain:
            success = await self.initialize_service()
            if not success:
                return {"success": False, "error": "Failed to initialize service"}
                
        try:
            # Extract request parameters
            prompt = request_data.get("prompt", "A scenic landscape")
            genre = request_data.get("genre", "cinematic")
            iterations = request_data.get("iterations", 1)
            negative_prompt = request_data.get("negative_prompt", "Bright tones, overexposed, static, blurred details, subtitles")
            guidance_scale = request_data.get("guidance_scale", 5)
            
            # Create the content request
            content_request = ContentRequest(
                prompt=prompt,
                genre=genre,
                iterations=iterations,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale
            )
            
            logger.info(f"Starting video generation with {iterations} iterations")
            start_time = time.time()
            
            # Call the WAN video generation function
            result = await self.story_chain.generate_video_WAN(content_request)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Process the result
            if "video_data" in result and "content_type" in result:
                # Save the video to the output directory
                timestamp = int(time.time())
                filename = f"wan_video_{iterations}iterations_{timestamp}.mp4"
                file_path = os.path.join(self.output_directory, filename)
                
                # Decode and save the video data
                video_data = base64.b64decode(result["video_data"])
                with open(file_path, "wb") as f:
                    f.write(video_data)
                
                logger.info(f"Successfully generated and saved video to {file_path}")
                logger.info(f"Total processing time: {total_time:.2f} seconds")
                
                # Return the result with metadata
                return {
                    "success": True,
                    "file_path": file_path,
                    "content_type": result["content_type"],
                    "processing_time": total_time,
                    "iterations": iterations,
                    "metrics": result.get("metrics", {})
                }
            else:
                logger.error("Failed to generate video: Invalid result format")
                return {
                    "success": False,
                    "error": "Invalid result format",
                    "result": result
                }
                
        except Exception as e:
            logger.error(f"Error in generate_video: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    async def process_multiple_requests(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple video generation requests sequentially
        
        Args:
            requests: List of request dictionaries
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        logger.info(f"Processing {len(requests)} video generation requests")
        
        for i, request in enumerate(requests):
            logger.info(f"\n{'='*20} Request {i+1}/{len(requests)} {'='*20}")
            logger.info(f"Processing request: {request}")
            
            # Generate the video
            result = await self.generate_video(request)
            results.append(result)
            
            # Log the result
            if result.get("success"):
                logger.info(f"Request {i+1}/{len(requests)} succeeded!")
                logger.info(f"Video saved to: {result.get('file_path')}")
                logger.info(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
            else:
                logger.error(f"Request {i+1}/{len(requests)} failed: {result.get('error')}")
                
            # Wait a bit between requests to avoid rate limiting
            if i < len(requests) - 1:
                logger.info("Waiting 5 seconds before next request...")
                await asyncio.sleep(5)
                
        return results
            
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        # Add any cleanup code here if needed

async def run_test_pipeline():
    """Run a test pipeline with different requests"""
    generator = WanVideoGenerator()
    
    # Define a list of test requests
    test_requests = [
        {
            "prompt": "A boat sailing on a serene lake at sunset with mountains in the background",
            "genre": "cinematic",
            "iterations": 1,
            "guidance_scale": 5
        },
        {
            "prompt": "A futuristic city at night with flying cars and neon lights",
            "genre": "sci-fi",
            "iterations": 2,
            "guidance_scale": 7
        },
        {
            "prompt": "Ancient ruins in a dense jungle with sunlight filtering through the canopy",
            "genre": "adventure",
            "iterations": 3,
            "guidance_scale": 5
        }
    ]
    
    # Process all requests
    results = await generator.process_multiple_requests(test_requests)
    
    # Summarize results
    logger.info("\n\n========== TEST PIPELINE SUMMARY ==========")
    success_count = sum(1 for r in results if r.get("success"))
    logger.info(f"Total requests: {len(results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(results) - success_count}")
    
    # Print details for each result
    for i, result in enumerate(results):
        status = "SUCCESS" if result.get("success") else "FAILED"
        logger.info(f"\nRequest {i+1}: {status}")
        
        if result.get("success"):
            logger.info(f"  File: {os.path.basename(result.get('file_path', ''))}")
            logger.info(f"  Time: {result.get('processing_time', 0):.2f} seconds")
            logger.info(f"  Iterations: {result.get('iterations', 0)}")
        else:
            logger.info(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Cleanup
    await generator.cleanup()
    
    return results

# Main function to run the test pipeline
async def main():
    try:
        logger.info("Starting WAN video generation test pipeline")
        results = await run_test_pipeline()
        logger.info("Test pipeline completed")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
