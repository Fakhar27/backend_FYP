"""
Hugging Face Inference API Service
Provides functions for text-to-image and text-to-video generation using Hugging Face models.
Uses direct API calls with polling for longer-running models.
"""

import os
import base64
import io
import time
import json
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime
from PIL import Image
import aiohttp
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)
load_dotenv()

class HuggingFaceService:
    """Service for interacting with Hugging Face Inference API using direct REST calls"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the HuggingFace service.
        
        Args:
            api_key: Optional API key. If not provided, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            logger.warning("No HUGGINGFACE_API_KEY provided or found in environment variables")
        
        # Create cache directory for temporary files
        self.temp_dir = "hf_temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info("HuggingFaceService initialized")
    
    async def generate_image_direct(self, prompt: str, model_id: str = "black-forest-labs/FLUX.1-schnell") -> Optional[str]:
        """
        Generate an image using direct Hugging Face Inference API REST calls with polling.
        
        Args:
            prompt: Text prompt for image generation
            model_id: Model ID to use for generation (default: FLUX.1-schnell)
            
        Returns:
            Base64-encoded image string or None if generation failed
        """
        try:
            start_time = time.time()
            logger.info(f"Starting direct image generation with model {model_id}")
            logger.info(f"Prompt: {prompt}")
            
            # Set up the API URL and headers
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Payload for the request
            payload = {"inputs": prompt}
            
            # Send initial request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)  # 2 minute timeout for initial request
                ) as response:
                    
                    # If we get a 200 response, we have the image directly
                    if response.status == 200:
                        logger.info("Received image response immediately")
                        image_bytes = await response.read()
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        
                        elapsed_time = time.time() - start_time
                        logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")
                        
                        return f"data:image/png;base64,{base64_image}"
                    
                    # If we get a 503, the model is loading and we need to poll
                    elif response.status == 503:
                        logger.info("Model is loading, starting to poll for results")
                        response_json = await response.json()
                        
                        # Check if there's an estimated time
                        estimated_time = response_json.get('estimated_time', 30)
                        logger.info(f"Estimated time: {estimated_time} seconds")
                        
                        # Poll for results
                        return await self._poll_for_result(api_url, headers, payload, start_time)
                    
                    # Handle other error codes
                    else:
                        error_text = await response.text()
                        logger.error(f"Error response ({response.status}): {error_text}")
                        return None
                
        except Exception as e:
            logger.error(f"Unexpected error in generate_image_direct: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    async def _poll_for_result(self, api_url: str, headers: Dict, payload: Dict, start_time: float) -> Optional[str]:
        """
        Poll for results from a long-running inference job.
        
        Args:
            api_url: API URL to poll
            headers: Request headers
            payload: Request payload
            start_time: Start time of the request for logging
            
        Returns:
            Base64-encoded result or None if failed
        """
        max_retries = 20  # Maximum number of polling attempts
        retry_delay = 5    # Initial delay between polls (seconds)
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                try:
                    # Wait before polling
                    logger.info(f"Waiting {retry_delay} seconds before polling (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    
                    # Send polling request
                    async with session.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        # Check if we got a result
                        if response.status == 200:
                            logger.info(f"Received successful response on poll attempt {attempt+1}")
                            image_bytes = await response.read()
                            base64_image = base64.b64encode(image_bytes).decode('utf-8')
                            
                            elapsed_time = time.time() - start_time
                            logger.info(f"Generation completed in {elapsed_time:.2f} seconds after {attempt+1} polling attempts")
                            
                            return f"data:image/png;base64,{base64_image}"
                        
                        # If still loading, continue polling
                        elif response.status == 503:
                            response_json = await response.json()
                            estimated_time = response_json.get('estimated_time', retry_delay)
                            logger.info(f"Model still loading. New estimated time: {estimated_time} seconds")
                            
                            # Adjust retry delay based on estimated time, but cap it
                            retry_delay = min(estimated_time, 15)
                            continue
                        
                        # Handle other errors
                        else:
                            error_text = await response.text()
                            logger.error(f"Error response during polling ({response.status}): {error_text}")
                            
                            # If we get a non-503 error, we should probably stop polling
                            if response.status != 503:
                                return None
                
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout during polling attempt {attempt+1}")
                    # Continue polling despite timeout
                    retry_delay = min(retry_delay * 1.5, 20)  # Increase delay, but cap it
                
                except Exception as e:
                    logger.error(f"Error during polling attempt {attempt+1}: {str(e)}")
                    return None
            
            # If we've exhausted all retries
            logger.error(f"Exceeded maximum polling attempts ({max_retries})")
            return None
    
    async def generate_video_direct(self, 
                                  prompt: str, 
                                  model_id: str = "cerspense/zeroscope_v2_XL") -> Optional[str]:
        """
        Generate a video using direct Hugging Face Inference API REST calls with polling.
        
        Args:
            prompt: Text prompt for video generation
            model_id: Model ID to use for generation
            
        Returns:
            Base64-encoded video string or None if generation failed
        """
        try:
            start_time = time.time()
            logger.info(f"Starting direct video generation with model {model_id}")
            logger.info(f"Prompt: {prompt}")
            
            # For video models, we typically need to use Replicate provider
            # We'll construct a different API endpoint based on the provider
            
            # For Replicate models (like zeroscope)
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Payload for video generation 
            payload = {
                "inputs": prompt,
                "parameters": {
                    "num_frames": 24,
                    "fps": 8
                }
            }
            
            # Initialize the generation and then poll for results
            async with aiohttp.ClientSession() as session:
                # Start the generation
                async with session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)  # 1 minute timeout for initial request
                ) as response:
                    
                    # If we get a 200 response immediately (unlikely for video)
                    if response.status == 200:
                        logger.info("Received video response immediately")
                        video_bytes = await response.read()
                        base64_video = base64.b64encode(video_bytes).decode('utf-8')
                        
                        elapsed_time = time.time() - start_time
                        logger.info(f"Video generation completed in {elapsed_time:.2f} seconds")
                        
                        return f"data:video/mp4;base64,{base64_video}"
                    
                    # If we get a 503, the model is loading and we need to poll
                    elif response.status == 503:
                        logger.info("Model is loading, starting to poll for results")
                        response_json = await response.json()
                        
                        # Poll for results - videos typically take longer
                        return await self._poll_for_result(api_url, headers, payload, start_time)
                    
                    # Handle other error codes
                    else:
                        error_text = await response.text()
                        logger.error(f"Error response ({response.status}): {error_text}")
                        return None
                    
        except Exception as e:
            logger.error(f"Unexpected error in generate_video_direct: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Cleaned up temporary directory")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# Utility functions

async def generate_image_with_hf(prompt: str, model_id: str = "black-forest-labs/FLUX.1-schnell") -> Optional[str]:
    """Utility function to generate an image without creating a service instance"""
    service = HuggingFaceService()
    try:
        return await service.generate_image_direct(prompt, model_id)
    finally:
        service.cleanup()

async def generate_video_with_hf(prompt: str, model_id: str = "cerspense/zeroscope_v2_XL") -> Optional[str]:
    """Utility function to generate a video without creating a service instance"""
    service = HuggingFaceService()
    try:
        return await service.generate_video_direct(prompt, model_id)
    finally:
        service.cleanup()


# Simple testing function
async def test_hf_service():
    """Test the HuggingFace service directly"""
    service = HuggingFaceService()
    try:
        # Test image generation
        image_result = await service.generate_image_direct("a peaceful lake with mountains in the background")
        if image_result:
            print("✅ Image generation successful")
            # Save image for testing
            img_data = base64.b64decode(image_result.split('base64,')[1])
            with open("test_image.png", "wb") as f:
                f.write(img_data)
            print("Image saved to test_image.png")
        else:
            print("❌ Image generation failed")
        
        # Test video generation
        video_result = await service.generate_video_direct("a peaceful lake with mountains in the background")
        if video_result:
            print("✅ Video generation successful")
            # Save video for testing
            video_data = base64.b64decode(video_result.split('base64,')[1])
            with open("test_video.mp4", "wb") as f:
                f.write(video_data)
            print("Video saved to test_video.mp4")
        else:
            print("❌ Video generation failed")
            
    finally:
        service.cleanup()

# Run the test if executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_hf_service())















# """
# Hugging Face Inference API Service
# Provides functions for text-to-image and text-to-video generation using Hugging Face models.
# """

# import os
# import base64
# import io
# import time
# import asyncio
# import logging
# from typing import Optional, Dict, Any, Tuple, Union
# from datetime import datetime
# from huggingface_hub import InferenceClient
# from PIL import Image
# import aiohttp
# from dotenv import load_dotenv

# # Import huggingface_hub for inference API access
# try:
#     from huggingface_hub import InferenceClient
# except ImportError:
#     raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

# # Set up logging
# logger = logging.getLogger(__name__)
# load_dotenv()

# class HuggingFaceService:
#     """Service for interacting with Hugging Face Inference API"""
    
#     def __init__(self, api_key: Optional[str] = None):
#         """
#         Initialize the HuggingFace service.
        
#         Args:
#             api_key: Optional API key. If not provided, will try to get from environment variable.
#         """
#         self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
#         if not self.api_key:
#             logger.warning("No HUGGINGFACE_API_KEY provided or found in environment variables")
        
#         # Create cache directory for temporary files
#         self.temp_dir = "hf_temp"
#         os.makedirs(self.temp_dir, exist_ok=True)
        
#         logger.info("HuggingFaceService initialized")
    
#     async def generate_image(self, prompt: str, model_id: str = "black-forest-labs/FLUX.1-schnell") -> Optional[str]:
#         """
#         Generate an image using Hugging Face Inference API.
        
#         Args:
#             prompt: Text prompt for image generation
#             model_id: Model ID to use for generation (default: FLUX.1-schnell)
            
#         Returns:
#             Base64-encoded image string or None if generation failed
#         """
#         try:
#             start_time = time.time()
#             logger.info(f"Starting image generation with model {model_id}")
#             logger.info(f"Prompt: {prompt}")
            
#             # Determine the provider based on the model
#             provider = "replicate" if "FLUX" in model_id or "stability" in model_id else "fal-ai"
#             logger.info(f"Using provider: {provider}")
            
#             # Create inference client
#             client = InferenceClient(
#                 provider=provider,
#                 api_key=self.api_key
#             )
            
#             # Generate image
#             try:
#                 image = client.text_to_image(
#                     prompt,
#                     model=model_id,
#                 )
                
#                 # Convert PIL image to base64
#                 buffered = io.BytesIO()
#                 image.save(buffered, format="PNG")
#                 img_str = base64.b64encode(buffered.getvalue()).decode()
#                 base64_image = f"data:image/png;base64,{img_str}"
                
#                 elapsed_time = time.time() - start_time
#                 logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")
                
#                 return base64_image
                
#             except Exception as e:
#                 logger.error(f"Error during image generation: {str(e)}")
#                 return None
                
#         except Exception as e:
#             logger.error(f"Unexpected error in generate_image: {str(e)}")
#             import traceback
#             logger.error(traceback.format_exc())
#             return None
    
#     async def generate_video(self, 
#                            prompt: str, 
#                            model_id: str = "cerspense/zeroscope_v2_XL", 
#                            num_frames: int = 24,
#                            fps: int = 8) -> Optional[str]:
#         """
#         Generate a video using Hugging Face Inference API.
        
#         Args:
#             prompt: Text prompt for video generation
#             model_id: Model ID to use for generation (default: Wan 2.1)
#             num_frames: Number of frames to generate (default: 24)
#             fps: Frames per second (default: 8)
            
#         Returns:
#             Base64-encoded video string or None if generation failed
#         """
#         try:
#             start_time = time.time()
#             logger.info(f"Starting video generation with model {model_id}")
#             logger.info(f"Prompt: {prompt}")
            
#             # Create inference client
#             # For text-to-video, we typically use Replicate
#             client = InferenceClient(
#                 provider="replicate",
#                 api_key=self.api_key
#             )
            
#             # Configure generation parameters
#             params = {
#                 "prompt": prompt,
#                 "num_frames": num_frames,
#                 "fps": fps
#             }
            
#             # Generate video
#             try:
#                 # For models that return direct video data
#                 response = client.post(
#                     model=model_id,
#                     data=params
#                 )
                
#                 # Most models return a URL to the generated video
#                 if isinstance(response, dict) and "output" in response:
#                     video_url = response["output"]
#                     logger.info(f"Video URL received: {video_url}")
                    
#                     # Download the video
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     video_path = os.path.join(self.temp_dir, f"generated_video_{timestamp}.mp4")
                    
#                     async with aiohttp.ClientSession() as session:
#                         async with session.get(video_url) as resp:
#                             if resp.status == 200:
#                                 with open(video_path, "wb") as f:
#                                     f.write(await resp.read())
                    
#                     # Convert to base64
#                     with open(video_path, "rb") as f:
#                         video_bytes = f.read()
                    
#                     video_base64 = base64.b64encode(video_bytes).decode("utf-8")
                    
#                     # Clean up
#                     try:
#                         os.remove(video_path)
#                     except:
#                         pass
                    
#                     elapsed_time = time.time() - start_time
#                     logger.info(f"Video generation completed in {elapsed_time:.2f} seconds")
                    
#                     return f"data:video/mp4;base64,{video_base64}"
#                 else:
#                     logger.error(f"Unexpected response format: {response}")
#                     return None
                    
#             except Exception as e:
#                 logger.error(f"Error during video generation: {str(e)}")
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 return None
                
#         except Exception as e:
#             logger.error(f"Unexpected error in generate_video: {str(e)}")
#             import traceback
#             logger.error(traceback.format_exc())
#             return None
    
#     def cleanup(self):
#         """Clean up temporary files"""
#         try:
#             import shutil
#             shutil.rmtree(self.temp_dir, ignore_errors=True)
#             logger.info("Cleaned up temporary directory")
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")


# # Utility functions for easy access

# async def generate_image_with_hf(prompt: str, model_id: str = "black-forest-labs/FLUX.1-schnell") -> Optional[str]:
#     """Utility function to generate an image without creating a service instance"""
#     service = HuggingFaceService()
#     try:
#         return await service.generate_image(prompt, model_id)
#     finally:
#         service.cleanup()

# async def generate_video_with_hf(prompt: str, model_id: str = "cerspense/zeroscope_v2_XL") -> Optional[str]:
#     """Utility function to generate a video without creating a service instance"""
#     service = HuggingFaceService()
#     try:
#         return await service.generate_video(prompt, model_id)
#     finally:
#         service.cleanup()


# # Example usage documentation

# """
# # Installation Requirements
# pip install huggingface_hub aiohttp Pillow python-dotenv

# # Environment Variables
# HUGGINGFACE_API_KEY=your_api_key_here

# # Example Usage
# async def example():
#     # For image generation
#     image_data = await generate_image_with_hf("Astronaut riding a horse on Mars")
    
#     # For video generation
#     video_data = await generate_video_with_hf("Astronaut riding a horse on Mars")
    
#     # Or using the service directly
#     hf_service = HuggingFaceService()
#     try:
#         image_data = await hf_service.generate_image("Astronaut riding a horse on Mars")
#         video_data = await hf_service.generate_video("Astronaut riding a horse on Mars")
#     finally:
#         hf_service.cleanup()
# """