import asyncio
import json
import os
import logging
import sys
import base64
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import the service and request model
from services.langchain_service import StoryIterationChain, ContentRequest

async def test_wan_video_generation():
    """
    Test function for the generate_video_WAN function.
    This simulates how the view function would call the service.
    """
    try:
        logger.info("Starting WAN video generation test")

        # Create a content request similar to how it's done in views.py
        prompt = "A boat sailing on a serene lake at sunset with mountains in the background"
        genre = "cinematic"
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        iterations = 2 
        guidance_scale = 5

        content_request = ContentRequest(
            prompt=prompt,
            genre=genre,
            iterations=iterations,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            backgroundVideo="1",
            backgroundMusic="1",
            voiceType="v2/en_speaker_6",
            subtitleColor="#ff00ff"
        )

        logger.info(f"Created content request: {content_request}")

        # Initialize the StoryIterationChain
        story_chain = StoryIterationChain()
        logger.info("Initialized StoryIterationChain")

        # Call the generate_video_WAN function
        logger.info("Calling generate_video_WAN...")
        result = await story_chain.generate_video_WAN(content_request)
        
        # Check if the result is successful
        if "video_data" in result and "content_type" in result:
            logger.info("Successfully generated video!")
            
            # Save the video data to a file
            video_data = base64.b64decode(result["video_data"])
            output_path = os.path.join(os.path.dirname(__file__), "test_output.mp4")
            
            with open(output_path, "wb") as f:
                f.write(video_data)
                
            logger.info(f"Video saved to: {output_path}")
            
            # Log metrics
            logger.info(f"Metrics: {result['metrics']}")
            
            return True
        else:
            logger.error(f"Failed to generate video: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error in test_wan_video_generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Main function to run the test
async def main():
    success = await test_wan_video_generation()
    if success:
        logger.info("Test completed successfully!")
    else:
        logger.error("Test failed!")

if __name__ == "__main__":
    asyncio.run(main())
