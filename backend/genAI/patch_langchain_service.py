import asyncio
import json
import os
import logging
import sys
import base64
import traceback
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define the function to patch the langchain_service.py file
def patch_langchain_service():
    """
    Patch the langchain_service.py file to fix the issue with data URLs
    """
    file_path = "E:/fyp_backend/backend/genAI/services/langchain_service.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the call_wan_api method
    start_marker = "async def call_wan_api(self, prompt: str, negative_prompt: str = \"\", guidance_scale: float = 5) -> str:"
    if start_marker not in content:
        logger.error("Could not find call_wan_api method in the file")
        return False
    
    # Find the video_url handling code
    url_check_marker = "video_url = response_data.get(\"video_url\")"
    if url_check_marker not in content:
        logger.error("Could not find video_url check in the file")
        return False
    
    # Replace the code after url_check_marker
    old_code = """            video_url = response_data.get("video_url")
            if not video_url:
                raise Exception("No video URL in response")
                
            # If the URL is relative, make it absolute
            if video_url.startswith("/"):
                video_url = f"https://api.deepinfra.com{video_url}"
                
            logger.info(f"WAN API returned video URL: {video_url}")
                
            # Download the video
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url, timeout=300) as video_response:
                    if video_response.status != 200:
                        raise Exception(f"Failed to download video: {video_response.status}")
                    
                    video_content = await video_response.read()"""
    
    new_code = """            video_url = response_data.get("video_url")
            if not video_url:
                raise Exception("No video URL in response")
            
            # Check if the URL is actually a data URL
            if video_url.startswith("data:"):
                logger.info("WAN API returned data URL")
                # Extract base64 data
                try:
                    # Format: data:video/mp4;base64,XXXXXXX
                    base64_data = video_url.split('base64,')[1]
                    # Return the base64 data directly
                    return base64_data
                except Exception as e:
                    logger.error(f"Error extracting base64 data: {str(e)}")
                    raise
                
            # If the URL is relative, make it absolute
            if video_url.startswith("/"):
                video_url = f"https://api.deepinfra.com{video_url}"
                
            logger.info(f"WAN API returned video URL: {video_url}")
                
            # Download the video
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url, timeout=300) as video_response:
                    if video_response.status != 200:
                        raise Exception(f"Failed to download video: {video_response.status}")
                    
                    video_content = await video_response.read()"""
    
    # Replace the code
    patched_content = content.replace(old_code, new_code)
    
    # Find the generate_video_WAN method processing of videos
    video_path_marker = "video_path = await self.call_wan_api("
    if video_path_marker not in content:
        logger.error("Could not find video_path code in the file")
        return False
    
    # Find the handling of video path in generate_video_WAN
    old_code_video_path = """                video_path = await self.call_wan_api(
                    prompt=content["image"],
                    negative_prompt=request.negative_prompt,
                    guidance_scale=request.guidance_scale
                )
                
                all_videos.append(video_path)"""
                
    new_code_video_path = """                result = await self.call_wan_api(
                    prompt=content["image"],
                    negative_prompt=request.negative_prompt,
                    guidance_scale=request.guidance_scale
                )
                
                # Check if result is a file path or base64 data
                if os.path.exists(result):
                    # It's a file path
                    logger.info(f"Result is a file path: {result}")
                    all_videos.append(result)
                else:
                    # It's base64 data
                    logger.info("Result is base64 data")
                    # Save it to a temporary file
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    temp_file.write(base64.b64decode(result))
                    temp_file.close()
                    logger.info(f"Saved base64 data to temporary file: {temp_file.name}")
                    all_videos.append(temp_file.name)"""
    
    # Replace the code
    patched_content = patched_content.replace(old_code_video_path, new_code_video_path)
    
    # Write the patched file
    with open(file_path + '.patched', 'w') as f:
        f.write(patched_content)
    
    logger.info(f"Patched file written to {file_path}.patched")
    return True

# Main function to run the patch
def main():
    logger.info("Starting patch process")
    success = patch_langchain_service()
    if success:
        logger.info("Patch successful!")
        logger.info("To apply the patch, run:")
        logger.info("mv E:/fyp_backend/backend/genAI/services/langchain_service.py.patched E:/fyp_backend/backend/genAI/services/langchain_service.py")
    else:
        logger.error("Patch failed")

if __name__ == "__main__":
    main()
