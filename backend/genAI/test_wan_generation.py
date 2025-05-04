import asyncio
import json
import os
import logging
import sys
import base64
import traceback
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import the service and request model
from services.langchain_service import StoryIterationChain, ContentRequest

class TestRunner:
    def __init__(self):
        self.story_chain = None
        
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
            
    async def run_wan_video_generation(self, prompt: str, iterations: int = 1, 
                                    guidance_scale: int = 5) -> Dict[str, Any]:
        """
        Run the WAN video generation with the given parameters
        
        Args:
            prompt: Text prompt for video generation
            iterations: Number of iterations to run (default: 1)
            guidance_scale: Guidance scale for generation (default: 5)
            
        Returns:
            Response dictionary with results
        """
        try:
            # Ensure service is initialized
            if not self.story_chain:
                success = await self.initialize_service()
                if not success:
                    return {"error": "Failed to initialize service"}
            
            # Create content request
            genre = "cinematic"
            negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
            
            content_request = ContentRequest(
                prompt=prompt,
                genre=genre,
                iterations=iterations,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale
            )
            
            logger.info(f"Created content request: {content_request}")
            
            # Call the generate_video_WAN function
            logger.info(f"Calling generate_video_WAN with prompt: '{prompt[:50]}...'")
            start_time = asyncio.get_event_loop().time()
            
            result = await self.story_chain.generate_video_WAN(content_request)
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            # Handle the result
            if "video_data" in result and "content_type" in result:
                logger.info(f"Successfully generated video in {duration:.2f} seconds!")
                
                # Save the video data to a file
                video_data = base64.b64decode(result["video_data"])
                output_path = os.path.join(os.path.dirname(__file__), f"wan_video_output_{iterations}iterations.mp4")
                
                with open(output_path, "wb") as f:
                    f.write(video_data)
                    
                logger.info(f"Video saved to: {output_path}")
                
                # Log metrics
                if "metrics" in result:
                    logger.info(f"Metrics: {result['metrics']}")
                
                return {
                    "success": True,
                    "video_path": output_path,
                    "duration": duration,
                    "metrics": result.get("metrics", {})
                }
            else:
                logger.error(f"Failed to generate video: {result}")
                return {
                    "success": False,
                    "error": "Video generation failed",
                    "result": result
                }
                
        except Exception as e:
            logger.error(f"Error in run_wan_video_generation: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    async def cleanup(self):
        """Clean up resources"""
        if self.story_chain:
            logger.info("Cleaning up resources...")
            # Add any cleanup code here if needed

async def test_different_prompts():
    """Test the WAN video generation with different prompts and settings"""
    runner = TestRunner()
    
    # Test case 1: Basic prompt with single iteration
    prompt1 = "A boat sailing on a serene lake at sunset with mountains in the background"
    logger.info(f"\n{'='*20} TEST CASE 1: Basic prompt with 1 iteration {'='*20}")
    result1 = await runner.run_wan_video_generation(prompt=prompt1, iterations=1)
    
    if result1.get("success"):
        logger.info(f"Test case 1 succeeded, video saved at: {result1.get('video_path')}")
    else:
        logger.error(f"Test case 1 failed: {result1.get('error')}")
    
    # Test case 2: Same prompt with multiple iterations
    if result1.get("success"):
        logger.info(f"\n{'='*20} TEST CASE 2: Same prompt with 2 iterations {'='*20}")
        result2 = await runner.run_wan_video_generation(prompt=prompt1, iterations=2)
        
        if result2.get("success"):
            logger.info(f"Test case 2 succeeded, video saved at: {result2.get('video_path')}")
        else:
            logger.error(f"Test case 2 failed: {result2.get('error')}")
    
    # Test case 3: Different prompt with different guidance scale
    logger.info(f"\n{'='*20} TEST CASE 3: Different prompt with guidance scale 7 {'='*20}")
    prompt3 = "A futuristic city at night with flying cars and neon lights"
    result3 = await runner.run_wan_video_generation(
        prompt=prompt3, 
        iterations=1,
        guidance_scale=7
    )
    
    if result3.get("success"):
        logger.info(f"Test case 3 succeeded, video saved at: {result3.get('video_path')}")
    else:
        logger.error(f"Test case 3 failed: {result3.get('error')}")
    
    # Cleanup
    await runner.cleanup()
    
    # Summary
    logger.info("\n\n========== TEST SUMMARY ==========")
    logger.info(f"Test case 1: {'SUCCESS' if result1.get('success') else 'FAILED'}")
    logger.info(f"Test case 2: {'SUCCESS' if 'result2' in locals() and result2.get('success') else 'FAILED or SKIPPED'}")
    logger.info(f"Test case 3: {'SUCCESS' if result3.get('success') else 'FAILED'}")

# Main function to run the tests
async def main():
    try:
        await test_different_prompts()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
