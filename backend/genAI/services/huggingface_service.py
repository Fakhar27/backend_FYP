"""
Hugging Face Inference API Service
Provides functions for text-to-image and text-to-video generation using Hugging Face models.
"""

import os
import base64
import io
import time
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime
from PIL import Image
import aiohttp
from dotenv import load_dotenv

# Import huggingface_hub for inference API access
try:
    from huggingface_hub import InferenceClient
except ImportError:
    raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

# Set up logging
logger = logging.getLogger(__name__)
load_dotenv()

class HuggingFaceService:
    """Service for interacting with Hugging Face Inference API"""
    
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
    
    async def generate_image(self, prompt: str, model_id: str = "black-forest-labs/FLUX.1-schnell") -> Optional[str]:
        """
        Generate an image using Hugging Face Inference API.
        
        Args:
            prompt: Text prompt for image generation
            model_id: Model ID to use for generation (default: FLUX.1-schnell)
            
        Returns:
            Base64-encoded image string or None if generation failed
        """
        try:
            start_time = time.time()
            logger.info(f"Starting image generation with model {model_id}")
            logger.info(f"Prompt: {prompt}")
            
            # Determine the provider based on the model
            provider = "replicate" if "FLUX" in model_id or "stability" in model_id else "fal-ai"
            logger.info(f"Using provider: {provider}")
            
            # Create inference client
            client = InferenceClient(
                provider=provider,
                api_key=self.api_key
            )
            
            # Generate image
            try:
                image = client.text_to_image(
                    prompt,
                    model=model_id,
                )
                
                # Convert PIL image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                base64_image = f"data:image/png;base64,{img_str}"
                
                elapsed_time = time.time() - start_time
                logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")
                
                return base64_image
                
            except Exception as e:
                logger.error(f"Error during image generation: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error in generate_image: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    async def generate_video(self, 
                           prompt: str, 
                           model_id: str = "cerspense/zeroscope_v2_XL", 
                           num_frames: int = 24,
                           fps: int = 8) -> Optional[str]:
        """
        Generate a video using Hugging Face Inference API.
        
        Args:
            prompt: Text prompt for video generation
            model_id: Model ID to use for generation (default: Wan 2.1)
            num_frames: Number of frames to generate (default: 24)
            fps: Frames per second (default: 8)
            
        Returns:
            Base64-encoded video string or None if generation failed
        """
        try:
            start_time = time.time()
            logger.info(f"Starting video generation with model {model_id}")
            logger.info(f"Prompt: {prompt}")
            
            # Create inference client
            # For text-to-video, we typically use Replicate
            client = InferenceClient(
                provider="replicate",
                api_key=self.api_key
            )
            
            # Configure generation parameters
            params = {
                "prompt": prompt,
                "num_frames": num_frames,
                "fps": fps
            }
            
            # Generate video
            try:
                # For models that return direct video data
                response = client.post(
                    model=model_id,
                    data=params
                )
                
                # Most models return a URL to the generated video
                if isinstance(response, dict) and "output" in response:
                    video_url = response["output"]
                    logger.info(f"Video URL received: {video_url}")
                    
                    # Download the video
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = os.path.join(self.temp_dir, f"generated_video_{timestamp}.mp4")
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(video_url) as resp:
                            if resp.status == 200:
                                with open(video_path, "wb") as f:
                                    f.write(await resp.read())
                    
                    # Convert to base64
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    
                    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
                    
                    # Clean up
                    try:
                        os.remove(video_path)
                    except:
                        pass
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"Video generation completed in {elapsed_time:.2f} seconds")
                    
                    return f"data:video/mp4;base64,{video_base64}"
                else:
                    logger.error(f"Unexpected response format: {response}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error during video generation: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error in generate_video: {str(e)}")
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


# Utility functions for easy access

async def generate_image_with_hf(prompt: str, model_id: str = "black-forest-labs/FLUX.1-schnell") -> Optional[str]:
    """Utility function to generate an image without creating a service instance"""
    service = HuggingFaceService()
    try:
        return await service.generate_image(prompt, model_id)
    finally:
        service.cleanup()

async def generate_video_with_hf(prompt: str, model_id: str = "cerspense/zeroscope_v2_XL") -> Optional[str]:
    """Utility function to generate a video without creating a service instance"""
    service = HuggingFaceService()
    try:
        return await service.generate_video(prompt, model_id)
    finally:
        service.cleanup()


# Example usage documentation

"""
# Installation Requirements
pip install huggingface_hub aiohttp Pillow python-dotenv

# Environment Variables
HUGGINGFACE_API_KEY=your_api_key_here

# Example Usage
async def example():
    # For image generation
    image_data = await generate_image_with_hf("Astronaut riding a horse on Mars")
    
    # For video generation
    video_data = await generate_video_with_hf("Astronaut riding a horse on Mars")
    
    # Or using the service directly
    hf_service = HuggingFaceService()
    try:
        image_data = await hf_service.generate_image("Astronaut riding a horse on Mars")
        video_data = await hf_service.generate_video("Astronaut riding a horse on Mars")
    finally:
        hf_service.cleanup()
"""