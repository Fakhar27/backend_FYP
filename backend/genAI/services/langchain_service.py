from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
import base64
from langchain_cohere.react_multi_hop.parsing import parse_answer_with_prefixes
from langchain_core.callbacks import BaseCallbackHandler
import tempfile
import time
import json
from langchain_core.outputs import LLMResult
from langsmith import Client
from langsmith.run_helpers import traceable, trace
import asyncio
from .aws_services import S3Handler
import aiohttp
import os
import logging
from typing import Optional, Dict, Any, List
from .video_manager import VideoManager
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

load_dotenv()

# class ContentRequest(BaseModel):
#     """Request model for story generation"""
#     prompt: str = Field(..., description="User's content prompt")
#     genre: str = Field(..., description="Content category/genre")
#     iterations: int = Field(default=4, ge=1, le=10)
class ContentRequest(BaseModel):
    """Request model for story generation"""
    prompt: str = Field(..., description="User's content prompt")
    genre: str = Field(..., description="Content category/genre")
    iterations: int = Field(default=3, ge=1, le=10)
    backgroundVideo: str = Field(default="1", description="Background video type")
    backgroundMusic: str = Field(default="1", description="Background music type")
    voiceType: str = Field(default="v2/en_speaker_6", description="Voice type (male or female)")
    subtitleColor: str = Field(default="#ff00ff", description="Subtitle text color")
    guidance_scale: int = Field(default=5)
    negative_prompt: str = Field(default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    # useHfInference: bool = Field(default=False, description="Whether to use Hugging Face Inference API")
    # hfImageModel: str = Field(default="black-forest-labs/FLUX.1-schnell", description="HF model for image generation")
    # useHfVideo: bool = Field(default=False, description="Whether to use Hugging Face for video generation")
    # hfVideoModel: str = Field(default="Wan-AI/Wan2.1-T2V-14B", description="HF model for video generation")

class ContentResponse(BaseModel):
    """Response model for each story iteration"""
    story: str
    image_description: str
    voice_data: Optional[str]
    image_url: Optional[str]
    iteration: int

class TokenUsageCallback(BaseCallbackHandler):
    """Callback handler to track token usage."""
    def __init__(self):
        super().__init__()
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.successful_requests = 0
        self.failed_requests = 0

    def on_llm_start(self, *args, **kwargs) -> None:
        """Called when LLM starts processing."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends processing."""
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.total_tokens += usage.get("total_tokens", 0)
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.successful_requests += 1
            logger.info(f"Token usage updated - Total: {self.total_tokens}")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM errors during processing."""
        self.failed_requests += 1
        logger.error(f"LLM error occurred: {str(error)}")

class StoryIterationChain:
    def __init__(self, colab_url: Optional[str] = None, voice_url: Optional[str] = None, whisper_url: Optional[str] = None):
        self.token_callback = TokenUsageCallback()
        self.client = Client()
        
        self.llm = ChatCohere(
            cohere_api_key=os.getenv("CO_API_KEY"),
            temperature=0.7,
            max_tokens=150,
            callbacks=[self.token_callback]
        )
        
        # self.colab_url = colab_url or os.getenv("COLAB_URL")
        # self.voice_url = voice_url or os.getenv("COLAB_URL_2")
        # self.whisper_url = whisper_url or os.getenv("COLAB_URL_3")
        self.voice_url = voice_url or os.getenv("COLAB_URL_2") or "https://4667-35-204-253-195.ngrok-free.app"
        self.whisper_url = whisper_url or os.getenv("COLAB_URL_3") or "https://8c5e-35-198-247-18.ngrok-free.app"
        
        # Colab URL for image generation is now optional
        self.colab_url = colab_url or os.getenv("COLAB_URL")
        
        # Initialize Hugging Face service if needed
        try:
            from .huggingface_service import HuggingFaceService
            self.hf_service = HuggingFaceService()
            self.hf_available = True
            logger.info("Hugging Face service initialized")
        except ImportError:
            self.hf_service = None
            self.hf_available = False
            logger.warning("Hugging Face service not available. Install with: pip install huggingface_hub")
            
        self.prefixes = {
            "story": "story:",
            "image": "image:"
        }
        
        self.base_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are generating very short story segments and image descriptions 
            in the {genre} genre.
            
            Format your response exactly as:
            story: [one sentence story]
            image: [detailed visual description]
            
            Requirements:
            - Keep story extremely brief (one sentence)
            - Make image descriptions specific and visual
            - Match the {genre} genre style and themes
            - Use exactly the format shown above"""),
            ("human", "{input_prompt}")
        ])
        
        self.continuation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Continue this {genre} story:
            Previous: {previous_story}
            
            Format your response exactly as:
            story: [one sentence continuation]
            image: [detailed visual description]
            
            Requirements:
            - Write only 1 sentence continuing the story
            - Keep image descriptions focused and specific
            - Match the {genre} genre style and themes
            - Use exactly the format shown above"""),
            ("human", "Continue the story.")
        ])

    @traceable(run_type="chain")
    async def generate_iteration(self, input_text: str, genre: str, previous_content: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Generate a single story iteration."""
        try:
            with trace(
                name="Story Generation Step",
                run_type="llm",
                project_name=os.getenv("LANGSMITH_PROJECT")
            ) as run:
                if previous_content is None:
                    prompt = self.base_prompt.format_prompt(
                        input_prompt=input_text,
                        genre=genre
                    )
                else:
                    prompt = self.continuation_prompt.format_prompt(
                        previous_story=previous_content["story"],
                        genre=genre
                    )
                
                response = await self.llm.ainvoke(
                    prompt.to_messages()
                )
                
                if isinstance(response.content, str):
                    parsed_content = parse_answer_with_prefixes(response.content, self.prefixes)
                else:
                    raise TypeError("Expected response.content to be a string")
                
                # Add run metadata
                run.add_metadata({
                    "token_usage": {
                        "total_tokens": self.token_callback.total_tokens,
                        "prompt_tokens": self.token_callback.prompt_tokens,
                        "completion_tokens": self.token_callback.completion_tokens
                    },
                    "request_stats": {
                        "successful": self.token_callback.successful_requests,
                        "failed": self.token_callback.failed_requests
                    }
                })
                
                return parsed_content
                
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            return {
                "story": "Error occurred in story generation.",
                "image": "Error occurred in image description."
            }
            
    async def generate_image(self, prompt: str, session: aiohttp.ClientSession) -> Optional[str]:
        """Generate image using Stable Diffusion API with retries"""
        if not self.colab_url:
            logger.error("COLAB_URL not set")
            return None
            
        retries = 3
        for attempt in range(retries):
            try:
                logger.info(f"Sending image generation request with prompt: {prompt}")
                
                async with session.post(
                    f"{self.colab_url}/generate-image",
                    json={"prompt": prompt},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if 'error' in result:
                        logger.error(f"Error from image generation: {result['error']}")
                        if attempt < retries - 1:
                            await asyncio.sleep(1)
                            continue
                        return None
                        
                    image_data = result.get('image_data')
                    if not image_data:
                        logger.error("No image data in response")
                        if attempt < retries - 1:
                            await asyncio.sleep(1)
                            continue
                        return None
                    
                    logger.info("Image generated successfully")
                    return image_data
                    
            except Exception as e:
                logger.error(f"Image generation failed (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None
                
        return None

    
    async def generate_voice(self, text: str, voice_type: str, session: aiohttp.ClientSession) -> Optional[str]:
        """Generate voice narration using Bark API with specific voice type"""
        if not self.voice_url:
            logger.error("Voice URL not set")
            return None
            
        retries = 3
        for attempt in range(retries):
            try:
                logger.info(f"Sending voice generation request for text: {text}, voice type: {voice_type}")
                
                async with session.post(
                    # f"https://4667-35-204-253-195.ngrok-free.app/generate_sound",
                    f"{self.voice_url}/generate_sound",
                    json={"text": text, "voice_type": voice_type},
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if 'error' in result:
                        logger.error(f"Error from voice generation: {result['error']}")
                        if attempt < retries - 1:
                            await asyncio.sleep(1)
                            continue
                        return None
                        
                    audio_data = result.get('audio_data')
                    if not audio_data:
                        logger.error("No audio data in response")
                        if attempt < retries - 1:
                            await asyncio.sleep(1)
                            continue
                        return None
                    
                    logger.info("Voice generated successfully")
                    return audio_data
                    
            except Exception as e:
                logger.error(f"Voice generation failed (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None
                
        return None
    
   
    async def call_wan_api(self, prompt: str, negative_prompt: str = "", guidance_scale: float = 5) -> str:
        """
        Call the Wan API to generate a video.
        
        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale
            
        Returns:
            Path to the saved video file or base64 data
        """
        try:
            # API endpoint and token
            api_url = "https://api.deepinfra.com/v1/inference/Wan-AI/Wan2.1-T2V-1.3B"
            api_token = os.getenv("DEEPINFRA_TOKEN")
            
            if not api_token:
                raise ValueError("DEEPINFRA_TOKEN environment variable not set")
            
            logger.info(f"Calling WAN API with prompt: '{prompt[:100]}...'")
            
            # Prepare request body
            request_body = {
                "prompt": prompt,
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt
            }
            
            # Make API request
            headers = {
                "Authorization": f"bearer {api_token}",
                "Content-Type": "application/json"
            }
            
            # Using aiohttp for async HTTP requests
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=request_body, headers=headers, timeout=600) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {response_text}")
                    
                    response_data = await response.json()
            
            # Get video URL from response
            video_url = response_data.get("video_url")
            if not video_url:
                raise Exception("No video URL in response")
            
            # Check if the URL is actually a data URL
            if video_url.startswith("data:"):
                logger.info("WAN API returned data URL")
                # Extract base64 data
                try:
                    # Format: data:video/mp4;base64,XXXXXXX
                    metadata, base64_data = video_url.split('base64,')
                    
                    # Write the decoded data to a temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    temp_file.write(base64.b64decode(base64_data))
                    temp_file.close()
                    
                    logger.info(f"Video data saved to temporary file: {temp_file.name}")
                    return temp_file.name
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
                    
                    video_content = await video_response.read()
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(video_content)
            temp_file.close()
            
            logger.info(f"Video saved to temporary file: {temp_file.name}")
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error calling WAN API: {str(e)}")
            raise
    
    
    @traceable(run_type="chain")
    async def generate_video_WAN_semi(self, request: ContentRequest) -> Dict[str, Any]:
        """
        Generate a video using the Wan API based on story segments.
        
        This function:
        1. Generates story segments and image descriptions using LLM
        2. For each segment, calls the Wan API to generate a video
        3. Concatenates all videos into a final video
        4. Returns the final video data
        """
        start_time = time.time()
        logger.info(f"Starting WAN video generation with prompt: '{request.prompt}'")
        
        # Track metrics
        metrics = {
            "start_time": start_time,
            "iterations": request.iterations,
        }
        
        try:
            # Initialize variables for tracking
            previous_content = None
            all_videos = []
            all_prompts = []
            
            # Generate videos for each iteration
            for i in range(request.iterations):
                iteration_start = time.time()
                logger.info(f"Starting iteration {i+1}/{request.iterations}")
                
                # Generate content for this iteration
                if i == 0:
                    content = await self.generate_iteration(request.prompt, request.genre)
                else:
                    content = await self.generate_iteration(request.prompt, request.genre, previous_content)
                
                # Save the content for the next iteration
                previous_content = content
                all_prompts.append(content)
                
                # Generate video using Wan API
                video_path = await self.call_wan_api(
                    prompt=content["image"],
                    negative_prompt=request.negative_prompt,
                    guidance_scale=request.guidance_scale
                )
                
                all_videos.append(video_path)
                logger.info(f"Completed iteration {i+1} in {time.time() - iteration_start:.2f}s")
            
            # Concatenate all videos
            from .video_manager import VideoManager
            video_manager = VideoManager()
            final_video_path = video_manager.concatenate_wan_videos(all_videos)
            
            # Read the final video
            with open(final_video_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Cleanup temporary files
            for video_path in all_videos:
                if os.path.exists(video_path):
                    os.remove(video_path)
            if os.path.exists(final_video_path):
                os.remove(final_video_path)
            
            # Calculate metrics
            end_time = time.time()
            metrics["total_duration"] = end_time - start_time
            metrics["prompts"] = all_prompts
            
            logger.info(f"Completed WAN video generation in {metrics['total_duration']:.2f}s")
            
            return {
                "video_data": video_data,
                "content_type": "video/mp4",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error in WAN video generation: {str(e)}")
            raise
    
    @traceable(run_type="chain")
    async def generate_video_WAN(self, request: ContentRequest) -> Dict[Any, Any]:
        """
        Generate a video using the Wan API based on story segments.
        
        This function:
        1. Generates story segments and image descriptions using LLM
        2. For each segment, calls the Wan API to generate a video
        3. Processes each video with audio narration and subtitles
        4. Concatenates all segments with background video and music
        5. Returns the final video data
        """
        start_time = time.time()
        logger.info(f"Starting WAN video generation with prompt: '{request.prompt}'")
        
        async with aiohttp.ClientSession() as session:
            with trace(
                name="WAN Video Generation Pipeline",
                run_type="chain",
                project_name=os.getenv("LANGSMITH_PROJECT")
            ) as run:
                video_manager = None
                s3_handler = None
                try:
                    # Ensure all required URLs are available
                    logger.info(f"Initializing pipeline with Whisper URL: {self.whisper_url}")
                    logger.info(f"Processing request with settings: genre={request.genre}, "
                            f"background={request.backgroundVideo}, music={request.backgroundMusic}, "
                            f"voice={request.voiceType}, color={request.subtitleColor}")
                    
                    if not self.whisper_url:
                        raise ValueError("Whisper URL is required")
                        
                    if not self.voice_url:
                        raise ValueError("Voice URL is required")
                    
                    # Initialize components
                    video_manager = VideoManager()
                    s3_handler = S3Handler()
                    segments_data = []
                    
                    # Track metrics
                    metrics = {
                        "start_time": start_time,
                        "iterations": request.iterations,
                        "prompts": []
                    }
                    
                    # Generate content for each iteration
                    previous_content = None
                    for i in range(request.iterations):
                        try:
                            logger.info(f"\n=== Processing Iteration {i + 1} ===")
                            
                            # Generate content for this iteration
                            if i == 0:
                                content = await self.generate_iteration(request.prompt, request.genre)
                            else:
                                content = await self.generate_iteration(request.prompt, request.genre, previous_content)
                            
                            # Save the content for the next iteration
                            previous_content = content
                            metrics["prompts"].append(content)
                            
                            # Extract story text for voice narration
                            story_text = content["story"]
                            logger.info(f"Generated story text: {story_text}")
                            
                            # Generate video using WAN API
                            logger.info(f"Generating video using WAN API with prompt: {content['image']}")
                            video_path = await self.call_wan_api(
                                prompt=content["image"],
                                negative_prompt=request.negative_prompt,
                                guidance_scale=request.guidance_scale
                            )
                            logger.info(f"Video generated and saved to: {video_path}")
                            
                            # Generate voice narration
                            logger.info(f"Generating voice narration")
                            audio_data = await self.generate_voice(
                                text=story_text,
                                voice_type=request.voiceType,
                                session=session
                            )
                            
                            if not audio_data:
                                raise ValueError(f"Failed to generate audio for iteration {i + 1}")
                            
                            # Convert video to base64 for segment creation
                            with open(video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                                video_data = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}"
                            
                            # Create segment with video and generated audio
                            segment_data = {
                                'video_path': video_path,
                                'audio_data': audio_data,
                                'story_text': story_text,
                                'subtitle_color': request.subtitleColor
                            }
                            
                            # Create segment with whisper for subtitles
                            # Note: VideoManager would need to be updated to handle videos instead of images
                            segment_path = await video_manager.create_video_segment(
                                segment_data,
                                i,
                                whisper_url=self.whisper_url,
                                session=session
                            )
                            
                            segments_data.append(segment_path)
                            
                            # Add metadata for tracing
                            run.add_metadata({
                                f"iteration_{i+1}": {
                                    "story": story_text,
                                    "prompt": content["image"],
                                    "status": "processed",
                                    "genre": request.genre
                                }
                            })
                            
                            logger.info(f"Completed iteration {i + 1}")
                            
                        except Exception as e:
                            logger.error(f"Error in iteration {i + 1}: {str(e)}")
                            raise ValueError(f"Failed in iteration {i + 1}: {str(e)}")
                    
                    # Get background video and music files from S3 based on user selection
                    background_video_path = s3_handler.get_media_file('video', request.backgroundVideo)
                    background_audio_path = s3_handler.get_media_file('music', request.backgroundMusic)
                    
                    logger.info(f"Selected background video: {background_video_path}")
                    logger.info(f"Selected background music: {background_audio_path}")
                    
                    # Fallback to hardcoded paths if S3 download fails
                    # if not background_video_path:
                    #     background_video_path = "E:/fyp_backend/backend/genAI/split_screen_video_1.mp4"
                    #     logger.warning(f"Using fallback video path: {background_video_path}")
                    
                    # if not background_audio_path:
                    #     background_audio_path = "E:/fyp_backend/backend/genAI/backgroundMusic1.wav"
                    #     logger.warning(f"Using fallback audio path: {background_audio_path}")
                    
                    # Concatenate video segments
                    logger.info("Starting video concatenation")
                    final_video_path = video_manager.concatenate_segments(
                        background_audio_path=background_audio_path,
                        split_video_path=background_video_path
                    )
                    
                    # Encode final video
                    logger.info("Encoding final video")
                    with open(final_video_path, 'rb') as video_file:
                        video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
                    
                    # Calculate metrics
                    end_time = time.time()
                    metrics["total_duration"] = end_time - start_time
                    
                    logger.info(f"Completed WAN video generation in {metrics['total_duration']:.2f}s")
                    
                    return {
                        "success": True,
                        "video_data": video_base64,
                        "content_type": "video/mp4",
                        "metrics": metrics
                    }
                    
                except Exception as e:
                    logger.error(f"Error in WAN video generation pipeline: {str(e)}")
                    raise
                
                finally:
                    if video_manager:
                        try:
                            video_manager.cleanup()
                        except Exception as e:
                            logger.error(f"Error during video manager cleanup: {str(e)}")
                    
                    if s3_handler:
                        try:
                            s3_handler.cleanup()
                        except Exception as e:
                            logger.error(f"Error during S3 handler cleanup: {str(e)}")                        
                            
    # WORKSSSSSSS
    @traceable(run_type="chain")
    async def generate_content_pipeline(self, request: ContentRequest) -> Dict[str, Any]:
        """Generate complete story with images and voice narration, return as video"""
        async with aiohttp.ClientSession() as session:
            with trace(
                name="Full Story Generation",
                run_type="chain",
                project_name=os.getenv("LANGSMITH_PROJECT")
            ) as run:
                video_manager = None
                s3_handler = None
                try:
                    logger.info(f"Initializing pipeline with Whisper URL: {self.whisper_url}")
                    logger.info(f"Processing request with settings: genre={request.genre}, "
                            f"background={request.backgroundVideo}, music={request.backgroundMusic}, "
                            f"voice={request.voiceType}, color={request.subtitleColor}")
                    
                    print(f"Using Whisper endpoint: {self.whisper_url}")
                    
                    if not self.whisper_url:
                        raise ValueError("Whisper URL is required")
                    
                    video_manager = VideoManager()
                    s3_handler = S3Handler()
                    previous_content = None
                    segments_data = []
                    
                    for i in range(request.iterations):
                        try:
                            print(f"\n=== Processing Iteration {i + 1} ===")
                            iteration_result = await self.generate_iteration(
                                input_text=request.prompt if i == 0 else "",
                                genre=request.genre,
                                previous_content=previous_content
                            )
                            image_task = asyncio.create_task(
                                self.generate_image(iteration_result["image"], session)
                            )
                            voice_task = asyncio.create_task(
                                self.generate_voice(
                                    text=iteration_result["story"], 
                                    voice_type=request.voiceType,
                                    session=session
                                )
                            )
                            image_data, audio_data = await asyncio.gather(
                                image_task,
                                voice_task,
                                return_exceptions=False 
                            )
                            
                            if not image_data or not audio_data:
                                raise ValueError(f"Failed to generate media for iteration {i + 1}")
                            segment_data = {
                                'image_data': image_data,
                                'audio_data': audio_data,
                                'story_text': iteration_result["story"],
                                'subtitle_color': request.subtitleColor
                            }
                            
                            segment_path = await video_manager.create_segment(
                                segment_data,
                                i,
                                whisper_url=self.whisper_url,
                                session=session
                            )
                            
                            previous_content = iteration_result
                            segments_data.append(segment_path)
                            
                            run.add_metadata({
                                f"iteration_{i+1}": {
                                    "story": iteration_result["story"],
                                    "image_description": iteration_result["image"],
                                    "status": "processed",
                                    "genre": request.genre
                                }
                            })
                            
                            logger.info(f"Completed iteration {i + 1}")
                            
                        except Exception as e:
                            logger.error(f"Error in iteration {i + 1}: {str(e)}")
                            raise ValueError(f"Failed in iteration {i + 1}: {str(e)}")
                    
                    # Get background video and music files from S3 based on user selection
                    background_video_path = s3_handler.get_media_file('video', request.backgroundVideo)
                    background_audio_path = s3_handler.get_media_file('music', request.backgroundMusic)
                    
                    logger.info(f"Selected background video: {background_video_path}")
                    logger.info(f"Selected background music: {background_audio_path}")
                    
                    # Fallback to hardcoded paths if S3 download fails
                    if not background_video_path:
                        background_video_path = "E:\\fyp_backend\\backend\\genAI\\split_screen_video_1.mp4"
                        logger.warning(f"Using fallback video path: {background_video_path}")
                    
                    if not background_audio_path:
                        background_audio_path = "E:\\fyp_backend\\backend\\genAI\\backgroundMusic1.wav"
                        logger.warning(f"Using fallback audio path: {background_audio_path}")
                    
                    logger.info("Starting video concatenation")
                    final_video_path = video_manager.concatenate_segments(
                        background_audio_path=background_audio_path,
                        split_video_path=background_video_path
                    )
                    
                    logger.info("Encoding final video")
                    with open(final_video_path, 'rb') as video_file:
                        video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
                    
                    return {
                        "success": True,
                        "video_data": video_base64,
                        "content_type": "video/mp4",
                        "metrics": {
                            "total_tokens": self.token_callback.total_tokens,
                            "successful_requests": self.token_callback.successful_requests,
                            "failed_requests": self.token_callback.failed_requests
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Error in video generation pipeline: {str(e)}")
                    raise
                
                finally:
                    if video_manager:
                        try:
                            video_manager.cleanup()
                        except Exception as e:
                            logger.error(f"Error during video manager cleanup: {str(e)}")
                    
                    if s3_handler:
                        try:
                            s3_handler.cleanup()
                        except Exception as e:
                            logger.error(f"Error during S3 handler cleanup: {str(e)}")
    
        
                            

# @traceable(run_type="chain")
#     async def generate_content_pipeline(self, request: ContentRequest) -> Dict[str, Any]:
#         """Generate complete story with images and voice narration, return as video"""
#         async with aiohttp.ClientSession() as session:
#             with trace(
#                 name="Full Story Generation",
#                 run_type="chain",
#                 project_name=os.getenv("LANGSMITH_PROJECT")
#             ) as run:
#                 video_manager = None
#                 try:
#                     logger.info(f"Initializing pipeline with Whisper URL: {self.whisper_url}")
#                     print(f"Using Whisper endpoint: {self.whisper_url}")
                    
#                     if not self.whisper_url:
#                         raise ValueError("Whisper URL is required")
                    
#                     video_manager = VideoManager()
#                     previous_content = None
#                     segments_data = []
                    
#                     for i in range(request.iterations):
#                         try:
#                             print(f"\n=== Processing Iteration {i + 1} ===")
#                             iteration_result = await self.generate_iteration(
#                                 input_text=request.prompt if i == 0 else "",
#                                 genre=request.genre,
#                                 previous_content=previous_content
#                             )
#                             image_task = asyncio.create_task(
#                                 self.generate_image(iteration_result["image"], session)
#                             )
#                             voice_task = asyncio.create_task(
#                                 self.generate_voice(iteration_result["story"], session)
#                             )
#                             image_data, audio_data = await asyncio.gather(
#                                 image_task,
#                                 voice_task,
#                                 return_exceptions=False 
#                             )
                            
#                             if not image_data or not audio_data:
#                                 raise ValueError(f"Failed to generate media for iteration {i + 1}")
#                             segment_data = {
#                                 'image_data': image_data,
#                                 'audio_data': audio_data,
#                                 'story_text': iteration_result["story"]
#                             }
                            
#                             segment_path = await video_manager.create_segment(
#                                 segment_data,
#                                 i,
#                                 whisper_url=self.whisper_url,
#                                 session=session
#                             )
                            
#                             previous_content = iteration_result
#                             segments_data.append(segment_path)
                            
#                             run.add_metadata({
#                                 f"iteration_{i+1}": {
#                                     "story": iteration_result["story"],
#                                     "image_description": iteration_result["image"],
#                                     "status": "processed",
#                                     "genre": request.genre
#                                 }
#                             })
                            
#                             logger.info(f"Completed iteration {i + 1}")
                            
#                         except Exception as e:
#                             logger.error(f"Error in iteration {i + 1}: {str(e)}")
#                             raise ValueError(f"Failed in iteration {i + 1}: {str(e)}")
                    
#                     logger.info("Starting video concatenation")
#                     final_video_path = video_manager.concatenate_segments(
#                         background_audio_path="E:\\fyp_backend\\backend\\genAI\\backgroundMusic1.wav",
#                         # split_video_path="E:\\fyp_backend\\backend\\genAI\\split_screen_video_1.mp4",
#                         )
                    
#                     logger.info("Encoding final video")
#                     with open(final_video_path, 'rb') as video_file:
#                         video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
                    
#                     return {
#                         "success": True,
#                         "video_data": video_base64,
#                         "content_type": "video/mp4",
#                         "metrics": {
#                             "total_tokens": self.token_callback.total_tokens,
#                             "successful_requests": self.token_callback.successful_requests,
#                             "failed_requests": self.token_callback.failed_requests
#                         }
#                     }
                    
#                 except Exception as e:
#                     logger.error(f"Error in video generation pipeline: {str(e)}")
#                     raise
                
#                 finally:
#                     if video_manager:
#                         try:
#                             video_manager.cleanup()
#                         except Exception as e:
#                             logger.error(f"Error during video manager cleanup: {str(e)}")
    
    
    # @traceable(run_type="chain")
    # async def generate_content_pipeline(self, request: ContentRequest) -> Dict[str, Any]:
    #     """Generate complete story with images and voice narration, return as video"""
    #     async with aiohttp.ClientSession() as session:
    #         with trace(
    #             name="Full Story Generation",
    #             run_type="chain",
    #             project_name=os.getenv("LANGSMITH_PROJECT")
    #         ) as run:
    #             video_manager = None
    #             try:
    #                 logger.info(f"Initializing pipeline with Whisper URL: {self.whisper_url}")
    #                 logger.info(f"Processing request with settings: genre={request.genre}, "
    #                         f"background={request.backgroundVideo}, music={request.backgroundMusic}, "
    #                         f"voice={request.voiceType}, color={request.subtitleColor}")
                    
    #                 print(f"Using Whisper endpoint: {self.whisper_url}")
                    
    #                 if not self.whisper_url:
    #                     raise ValueError("Whisper URL is required")
                    
    #                 video_manager = VideoManager()
    #                 previous_content = None
    #                 segments_data = []
                    
    #                 for i in range(request.iterations):
    #                     try:
    #                         print(f"\n=== Processing Iteration {i + 1} ===")
    #                         iteration_result = await self.generate_iteration(
    #                             input_text=request.prompt if i == 0 else "",
    #                             genre=request.genre,
    #                             previous_content=previous_content
    #                         )
    #                         image_task = asyncio.create_task(
    #                             self.generate_image(iteration_result["image"], session)
    #                         )
    #                         voice_task = asyncio.create_task(
    #                             self.generate_voice(
    #                                 text=iteration_result["story"], 
    #                                 voice_type=request.voiceType,  # Pass voice type to the generator
    #                                 session=session
    #                             )
    #                         )
    #                         image_data, audio_data = await asyncio.gather(
    #                             image_task,
    #                             voice_task,
    #                             return_exceptions=False 
    #                         )
                            
    #                         if not image_data or not audio_data:
    #                             raise ValueError(f"Failed to generate media for iteration {i + 1}")
    #                         segment_data = {
    #                             'image_data': image_data,
    #                             'audio_data': audio_data,
    #                             'story_text': iteration_result["story"],
    #                             'subtitle_color': request.subtitleColor  # Pass subtitle color
    #                         }
                            
    #                         segment_path = await video_manager.create_segment(
    #                             segment_data,
    #                             i,
    #                             whisper_url=self.whisper_url,
    #                             session=session
    #                         )
                            
    #                         previous_content = iteration_result
    #                         segments_data.append(segment_path)
                            
    #                         run.add_metadata({
    #                             f"iteration_{i+1}": {
    #                                 "story": iteration_result["story"],
    #                                 "image_description": iteration_result["image"],
    #                                 "status": "processed",
    #                                 "genre": request.genre
    #                             }
    #                         })
                            
    #                         logger.info(f"Completed iteration {i + 1}")
                            
    #                     except Exception as e:
    #                         logger.error(f"Error in iteration {i + 1}: {str(e)}")
    #                         raise ValueError(f"Failed in iteration {i + 1}: {str(e)}")
                    
    #                 # TODO: In the future, these paths should come from a database based on the request parameters
    #                 # For now, we'll keep the hardcoded paths but log what would be selected
    #                 # logger.info(f"Would select background video: {request.backgroundVideo}")
    #                 # logger.info(f"Would select background music: {request.backgroundMusic}")
                    
    #                 # Current hardcoded paths
    #                 # background_audio_path = this.background_audio_path 
    #                 background_audio_path = "E:\\fyp_backend\\backend\\genAI\\backgroundMusic1.wav"
    #                 split_video_path = "E:\\fyp_backend\\backend\\genAI\\split_screen_video_1.mp4"
                    
    #                 logger.info("Starting video concatenation")
    #                 final_video_path = video_manager.concatenate_segments(
    #                     background_audio_path=background_audio_path,
    #                     split_video_path=split_video_path
    #                 )
                    
    #                 logger.info("Encoding final video")
    #                 with open(final_video_path, 'rb') as video_file:
    #                     video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
                    
    #                 return {
    #                     "success": True,
    #                     "video_data": video_base64,
    #                     "content_type": "video/mp4",
    #                     "metrics": {
    #                         "total_tokens": self.token_callback.total_tokens,
    #                         "successful_requests": self.token_callback.successful_requests,
    #                         "failed_requests": self.token_callback.failed_requests
    #                     }
    #                 }
                    
    #             except Exception as e:
    #                 logger.error(f"Error in video generation pipeline: {str(e)}")
    #                 raise
                
    #             finally:
    #                 if video_manager:
    #                     try:
    #                         video_manager.cleanup()
    #                     except Exception as e:
    #                         logger.error(f"Error during video manager cleanup: {str(e)}")
    
    
    

# async def generate_voice(self, text: str, session: aiohttp.ClientSession) -> Optional[str]:
#         """Generate voice narration using Bark API"""
#         if not self.voice_url:
#             logger.error("Voice URL not set")
#             return None
            
#         retries = 3
#         for attempt in range(retries):
#             try:
#                 logger.info(f"Sending voice generation request for text: {text}")
                
#                 async with session.post(
#                     f"{self.voice_url}/generate_sound",
#                     json={"text": text},
#                     timeout=aiohttp.ClientTimeout(total=300)
#                 ) as response:
#                     response.raise_for_status()
#                     result = await response.json()
                    
#                     if 'error' in result:
#                         logger.error(f"Error from voice generation: {result['error']}")
#                         if attempt < retries - 1:
#                             await asyncio.sleep(1)
#                             continue
#                         return None
                        
#                     audio_data = result.get('audio_data')
#                     if not audio_data:
#                         logger.error("No audio data in response")
#                         if attempt < retries - 1:
#                             await asyncio.sleep(1)
#                             continue
#                         return None
                    
#                     logger.info("Voice generated successfully")
#                     return audio_data
                    
#             except Exception as e:
#                 logger.error(f"Voice generation failed (attempt {attempt + 1}/{retries}): {str(e)}")
#                 if attempt < retries - 1:
#                     await asyncio.sleep(1)
#                     continue
#                 return None
                
#         return None
    
    
    
    
    
    
    
    
    # @traceable(run_type="chain")
    # async def generate_content_pipeline(self, request: ContentRequest) -> Dict[str, Any]:
    #     """Generate complete story with images and voice narration, return as video"""
    #     async with aiohttp.ClientSession() as session:
    #         with trace(
    #             name="Full Story Generation",
    #             run_type="chain",
    #             project_name=os.getenv("LANGSMITH_PROJECT")
    #         ) as run:
    #             video_manager = None
    #             s3_handler = None
    #             try:
    #                 logger.info(f"Initializing pipeline with Whisper URL: {self.whisper_url}")
    #                 logger.info(f"Processing request with settings: genre={request.genre}, "
    #                         f"background={request.backgroundVideo}, music={request.backgroundMusic}, "
    #                         f"voice={request.voiceType}, color={request.subtitleColor}")
    #                 logger.info(f"Using Hugging Face: {request.useHfInference}")
                    
    #                 if not self.whisper_url:
    #                     raise ValueError("Whisper URL is required")
                    
    #                 if not self.voice_url:
    #                     raise ValueError("Voice URL is required")
                    
    #                 # Check if we're using Hugging Face and if it's available
    #                 if request.useHfInference and not self.hf_available:
    #                     logger.error("Hugging Face service requested but not available")
    #                     raise ValueError("Hugging Face service is not available. Please install huggingface_hub.")
                    
    #                 # If using normal Colab image generation, make sure the URL is available
    #                 if not request.useHfInference and not self.colab_url:
    #                     logger.error("Colab URL is required for image generation but not provided")
    #                     raise ValueError("Colab URL is required for image generation")
                    
    #                 video_manager = VideoManager()
    #                 s3_handler = S3Handler()
    #                 previous_content = None
    #                 segments_data = []
                    
    #                 # We can also attempt direct text-to-video generation if requested
    #                 if request.useHfVideo and self.hf_available:
    #                     logger.info(f"Using Hugging Face for direct text-to-video generation with model {request.hfVideoModel}")
    #                     try:
    #                         # Generate the video directly with Hugging Face
    #                         video_data = await self.hf_service.generate_video(
    #                             prompt=request.prompt,
    #                             model_id=request.hfVideoModel
    #                         )
                            
    #                         if not video_data:
    #                             logger.error("Failed to generate video with HF, falling back to standard pipeline")
    #                         else:
    #                             # We got a video! Return it directly
    #                             logger.info("Successfully generated video with Hugging Face")
    #                             return {
    #                                 "success": True,
    #                                 "video_data": video_data.split("base64,")[1] if "base64," in video_data else video_data,
    #                                 "content_type": "video/mp4",
    #                                 "metrics": {
    #                                     "total_tokens": self.token_callback.total_tokens,
    #                                     "successful_requests": self.token_callback.successful_requests,
    #                                     "failed_requests": self.token_callback.failed_requests,
    #                                     "method": "huggingface_video_direct"
    #                                 }
    #                             }
    #                     except Exception as e:
    #                         logger.error(f"Error in HF video generation: {str(e)}")
    #                         logger.info("Falling back to standard pipeline")
                    
    #                 # Standard iterative pipeline
    #                 for i in range(request.iterations):
    #                     try:
    #                         print(f"\n=== Processing Iteration {i + 1} ===")
    #                         iteration_result = await self.generate_iteration(
    #                             input_text=request.prompt if i == 0 else "",
    #                             genre=request.genre,
    #                             previous_content=previous_content
    #                         )
                            
    #                         # Choose image generation method based on configuration
    #                         if request.useHfInference and self.hf_available:
    #                             logger.info(f"Using Hugging Face for image generation with model {request.hfImageModel}")
    #                             image_task = asyncio.create_task(
    #                                 self.hf_service.generate_image(
    #                                     prompt=iteration_result["image"],
    #                                     model_id=request.hfImageModel
    #                                 )
    #                             )
    #                         else:
    #                             logger.info(f"Using Colab for image generation at URL: {self.colab_url}")
    #                             image_task = asyncio.create_task(
    #                                 self.generate_image(iteration_result["image"], session)
    #                             )
                                
    #                         voice_task = asyncio.create_task(
    #                             self.generate_voice(
    #                                 text=iteration_result["story"], 
    #                                 voice_type=request.voiceType,
    #                                 session=session
    #                             )
    #                         )
                            
    #                         image_data, audio_data = await asyncio.gather(
    #                             image_task,
    #                             voice_task,
    #                             return_exceptions=False 
    #                         )
                            
    #                         if not image_data or not audio_data:
    #                             raise ValueError(f"Failed to generate media for iteration {i + 1}")
                            
    #                         segment_data = {
    #                             'image_data': image_data,
    #                             'audio_data': audio_data,
    #                             'story_text': iteration_result["story"],
    #                             'subtitle_color': request.subtitleColor
    #                         }
                            
    #                         segment_path = await video_manager.create_segment(
    #                             segment_data,
    #                             i,
    #                             whisper_url=self.whisper_url,
    #                             session=session
    #                         )
                            
    #                         previous_content = iteration_result
    #                         segments_data.append(segment_path)
                            
    #                         run.add_metadata({
    #                             f"iteration_{i+1}": {
    #                                 "story": iteration_result["story"],
    #                                 "image_description": iteration_result["image"],
    #                                 "status": "processed",
    #                                 "genre": request.genre
    #                             }
    #                         })
                            
    #                         logger.info(f"Completed iteration {i + 1}")
                            
    #                     except Exception as e:
    #                         logger.error(f"Error in iteration {i + 1}: {str(e)}")
    #                         raise ValueError(f"Failed in iteration {i + 1}: {str(e)}")
                    
    #                 # Get background video and music files from S3 based on user selection
    #                 background_video_path = s3_handler.get_media_file('video', request.backgroundVideo)
    #                 background_audio_path = s3_handler.get_media_file('music', request.backgroundMusic)
                    
    #                 logger.info(f"Selected background video: {background_video_path}")
    #                 logger.info(f"Selected background music: {background_audio_path}")
                    
    #                 # Fallback to hardcoded paths if S3 download fails
    #                 if not background_video_path or background_video_path == "NONE":
    #                     background_video_path = "NONE" if background_video_path == "NONE" else "E:\\fyp_backend\\backend\\genAI\\split_screen_video_1.mp4"
    #                     logger.warning(f"Using path: {background_video_path}")
                    
    #                 if not background_audio_path or background_audio_path == "NONE":
    #                     background_audio_path = "NONE" if background_audio_path == "NONE" else "E:\\fyp_backend\\backend\\genAI\\backgroundMusic1.wav"
    #                     logger.warning(f"Using path: {background_audio_path}")
                    
    #                 logger.info("Starting video concatenation")
    #                 final_video_path = video_manager.concatenate_segments(
    #                     background_audio_path=background_audio_path,
    #                     split_video_path=background_video_path
    #                 )
                    
    #                 logger.info("Encoding final video")
    #                 with open(final_video_path, 'rb') as video_file:
    #                     video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
                    
    #                 return {
    #                     "success": True,
    #                     "video_data": video_base64,
    #                     "content_type": "video/mp4",
    #                     "metrics": {
    #                         "total_tokens": self.token_callback.total_tokens,
    #                         "successful_requests": self.token_callback.successful_requests,
    #                         "failed_requests": self.token_callback.failed_requests,
    #                         "method": "iterative_pipeline"
    #                     }
    #                 }
                    
    #             except Exception as e:
    #                 logger.error(f"Error in video generation pipeline: {str(e)}")
    #                 import traceback
    #                 logger.error(traceback.format_exc())
    #                 raise
                
    #             finally:
    #                 if video_manager:
    #                     try:
    #                         video_manager.cleanup()
    #                     except Exception as e:
    #                         logger.error(f"Error during video manager cleanup: {str(e)}")
                    
    #                 if s3_handler:
    #                     try:
    #                         s3_handler.cleanup()
    #                     except Exception as e:
    #                         logger.error(f"Error during S3 handler cleanup: {str(e)}")
                            
    #                 if hasattr(self, 'hf_service') and self.hf_service:
    #                     try:
    #                         self.hf_service.cleanup()
    #                     except Exception as e:
    #                         logger.error(f"Error during HF service cleanup: {str(e)}")
                            
                            
                            
                            
                            
                             # async def call_wan_api(self, prompt: str, negative_prompt: str = "", guidance_scale: float = 5) -> str:
    #     """
    #     Call the Wan API to generate a video.
        
    #     Args:
    #         prompt: Text prompt for video generation
    #         negative_prompt: Negative prompt
    #         guidance_scale: Guidance scale
            
    #     Returns:
    #         Path to the saved video file
    #     """
    #     try:
    #         # API endpoint and token
    #         api_url = "https://api.deepinfra.com/v1/inference/Wan-AI/Wan2.1-T2V-1.3B"
    #         api_token = os.getenv("DEEPINFRA_TOKEN")
            
    #         if not api_token:
    #             raise ValueError("DEEPINFRA_TOKEN environment variable not set")
            
    #         logger.info(f"Calling WAN API with prompt: '{prompt[:100]}...'")
            
    #         # Prepare request body
    #         request_body = {
    #             "prompt": prompt,
    #             "guidance_scale": guidance_scale,
    #             "negative_prompt":negative_prompt
    #         }
            
            
    #         # Make API request
    #         headers = {
    #             "Authorization": f"bearer {api_token}",
    #             "Content-Type": "application/json"
    #         }
            
    #         # Using aiohttp for async HTTP requests
    #         async with aiohttp.ClientSession() as session:
    #             async with session.post(api_url, json=request_body, headers=headers, timeout=600) as response:
    #                 if response.status != 200:
    #                     response_text = await response.text()
    #                     raise Exception(f"API request failed with status {response.status}: {response_text}")
                    
    #                 response_data = await response.json()
            
    #         # Get video URL from response
    #         video_url = response_data.get("video_url")
    #         if not video_url:
    #             raise Exception("No video URL in response")
                
    #         # If the URL is relative, make it absolute
    #         if video_url.startswith("/"):
    #             video_url = f"https://api.deepinfra.com{video_url}"
                
    #         logger.info(f"WAN API returned video URL: {video_url}")
                
    #         # Download the video
    #         async with aiohttp.ClientSession() as session:
    #             async with session.get(video_url, timeout=300) as video_response:
    #                 if video_response.status != 200:
    #                     raise Exception(f"Failed to download video: {video_response.status}")
                    
    #                 video_content = await video_response.read()
            
    #         # Save to temporary file
    #         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    #         temp_file.write(video_content)
    #         temp_file.close()
            
    #         logger.info(f"Video saved to temporary file: {temp_file.name}")
            
    #         return temp_file.name
            
    #     except Exception as e:
    #         logger.error(f"Error calling WAN API: {str(e)}")
    #         raise
    
    
    
    
    
    
    
    
    # async def generate_image_hf(self, prompt: str, model_id: str = "black-forest-labs/FLUX.1-schnell") -> Optional[str]:
    #     """Generate image using Hugging Face API with direct API calls"""
    #     try:
    #         logger.info(f"Using HuggingFace direct API for image generation with model: {model_id}")
    #         logger.info(f"Prompt: {prompt}")

    #         # Import here to avoid circular imports (as in your original code)
    #         # NOTE: This assumes huggingface_service contains the actual implementation
    #         from .huggingface_service import generate_image_with_hf

    #         # Call the direct API function (assuming it's correctly implemented in huggingface_service)
    #         image_data = await generate_image_with_hf(prompt, model_id)

    #         if not image_data:
    #             logger.error("No image data returned from Hugging Face")
    #             return None

    #         logger.info("Successfully generated image with Hugging Face")
    #         return image_data

    #     except Exception as e:
    #         logger.error(f"Error in HF image generation: {str(e)}")
    #         # logger.error(traceback.format_exc())
    #         return None


    # async def generate_video_hf(self, prompt: str, model_id: str = "cerspense/zeroscope_v2_XL") -> Optional[str]:
    #     """Generate video directly using Hugging Face API"""
    #     try:
    #         logger.info(f"Using HuggingFace direct API for video generation with model: {model_id}")
    #         logger.info(f"Prompt: {prompt}")

    #         # Import here to avoid circular imports (as in your original code)
    #         # NOTE: This assumes huggingface_service contains the actual implementation
    #         from .huggingface_service import generate_video_with_hf

    #         # Call the direct API function (assuming it's correctly implemented in huggingface_service)
    #         video_data = await generate_video_with_hf(prompt, model_id)

    #         if not video_data:
    #             logger.error("No video data returned from Hugging Face")
    #             return None

    #         logger.info("Successfully generated video with Hugging Face")
    #         return video_data

    #     except Exception as e:
    #         logger.error(f"Error in HF video generation: {str(e)}")
    #         # logger.error(traceback.format_exc())
    #         return None
    
    
    
    
    
    
     # @traceable(run_type="chain")
    # async def generate_content_pipeline(self, request: ContentRequest) -> Dict[str, Any]:
    #     """Generate complete story with hardcoded images and stories"""
    #     async with aiohttp.ClientSession() as session:
    #         with trace(
    #             name="Full Story Generation",
    #             run_type="chain",
    #             project_name=os.getenv("LANGSMITH_PROJECT")
    #         ) as run:
    #             video_manager = None
    #             s3_handler = None
    #             try:
    #                 # Ensure all required URLs are available except Colab URL
    #                 logger.info(f"Initializing pipeline with Whisper URL: {self.whisper_url}")
    #                 logger.info(f"Processing request with settings: genre={request.genre}, "
    #                         f"background={request.backgroundVideo}, music={request.backgroundMusic}, "
    #                         f"voice={request.voiceType}, color={request.subtitleColor}")
                    
    #                 print(f"Using Whisper endpoint: {self.whisper_url}")
                    
    #                 if not self.whisper_url:
    #                     raise ValueError("Whisper URL is required")
                        
    #                 if not self.voice_url:
    #                     raise ValueError("Voice URL is required")
                    
    #                 # Get hardcoded story prompts and image paths
    #                 hardcoded_story_prompts = [
    #                     "A lone traveler journeyed across vast golden dunes following mysterious footprints, his flowing robes catching the desert wind as he pursued his unknown destiny.",
    #                     "At sunset, he discovered a perfect circular oasis with reflecting palms offering refuge and a sign to continue his quest beyond this desert sanctuary.",
    #                     "Venturing further, he found where desert transformed into paradise, wading through turquoise waters between tall palms as golden light filtered through the lush canopy.",
    #                     "In a hidden clearing, he sat beside a small pool, his touch creating ripples that awakened golden symbols above an ancient tree, revealing the sacred connection between human intention and nature's magic."
    #                 ]
                    
    #                 hardcoded_image_paths = [
    #                     "E:/fyp_backend/backend/genAI/image_man_in_desert_1.webp",
    #                     "E:/fyp_backend/backend/genAI/image_man_in_desert_2.webp",
    #                     "E:/fyp_backend/backend/genAI/image_man_in_desert_3.webp",
    #                     "E:/fyp_backend/backend/genAI/image_man_in_desert_4.webp"
    #                 ]
                    
    #                 # Limit the iterations to the number of hardcoded items we have
    #                 iterations = min(request.iterations, len(hardcoded_story_prompts))
    #                 logger.info(f"Using {iterations} hardcoded iterations")
                    
    #                 video_manager = VideoManager()
    #                 s3_handler = S3Handler()
    #                 segments_data = []
                    
    #                 # Process each iteration with hardcoded content
    #                 for i in range(iterations):
    #                     try:
    #                         logger.info(f"\n=== Processing Iteration {i + 1} ===")
                            
    #                         # Get hardcoded story for this iteration
    #                         story_text = hardcoded_story_prompts[i]
    #                         logger.info(f"Using hardcoded story: {story_text}")
                            
    #                         # Read hardcoded image file
    #                         image_path = hardcoded_image_paths[i]
    #                         logger.info(f"Using hardcoded image: {image_path}")
                            
    #                         # Convert image file to base64
    #                         try:
    #                             with open(image_path, 'rb') as image_file:
    #                                 image_bytes = image_file.read()
    #                                 image_data = f"data:image/webp;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    #                                 logger.info(f"Successfully loaded hardcoded image {i+1}")
    #                         except Exception as e:
    #                             logger.error(f"Error loading hardcoded image: {str(e)}")
    #                             raise ValueError(f"Failed to load hardcoded image for iteration {i + 1}")
                            
    #                         # Generate voice narration using the actual service
    #                         logger.info(f"Generating voice narration for hardcoded story {i+1}")
    #                         audio_data = await self.generate_voice(
    #                             text=story_text,
    #                             voice_type=request.voiceType,
    #                             session=session
    #                         )
                            
    #                         if not audio_data:
    #                             raise ValueError(f"Failed to generate audio for iteration {i + 1}")
                            
    #                         # Create segment with hardcoded image and generated audio
    #                         segment_data = {
    #                             'image_data': image_data,
    #                             'audio_data': audio_data,
    #                             'story_text': story_text,
    #                             'subtitle_color': request.subtitleColor
    #                         }
                            
    #                         # Create segment with whisper for subtitles
    #                         segment_path = await video_manager.create_segment(
    #                             segment_data,
    #                             i,
    #                             whisper_url=self.whisper_url,
    #                             session=session
    #                         )
                            
    #                         segments_data.append(segment_path)
                            
    #                         # Add metadata for tracing
    #                         run.add_metadata({
    #                             f"iteration_{i+1}": {
    #                                 "story": story_text,
    #                                 "image_path": image_path,
    #                                 "status": "processed",
    #                                 "genre": request.genre
    #                             }
    #                         })
                            
    #                         logger.info(f"Completed iteration {i + 1}")
                            
    #                     except Exception as e:
    #                         logger.error(f"Error in iteration {i + 1}: {str(e)}")
    #                         raise ValueError(f"Failed in iteration {i + 1}: {str(e)}")
                    
    #                 # Get background video and music files from S3 based on user selection
    #                 background_video_path = s3_handler.get_media_file('video', request.backgroundVideo)
    #                 background_audio_path = s3_handler.get_media_file('music', request.backgroundMusic)
                    
    #                 logger.info(f"Selected background video: {background_video_path}")
    #                 logger.info(f"Selected background music: {background_audio_path}")
                    
    #                 # Fallback to hardcoded paths if S3 download fails
    #                 if not background_video_path:
    #                     background_video_path = "E:/fyp_backend/backend/genAI/split_screen_video_1.mp4"
    #                     logger.warning(f"Using fallback video path: {background_video_path}")
                    
    #                 if not background_audio_path:
    #                     background_audio_path = "E:/fyp_backend/backend/genAI/backgroundMusic1.wav"
    #                     logger.warning(f"Using fallback audio path: {background_audio_path}")
                    
    #                 # Concatenate video segments
    #                 logger.info("Starting video concatenation")
    #                 final_video_path = video_manager.concatenate_segments(
    #                     background_audio_path=background_audio_path,
    #                     split_video_path=background_video_path
    #                 )
                    
    #                 # Encode final video
    #                 logger.info("Encoding final video")
    #                 with open(final_video_path, 'rb') as video_file:
    #                     video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
                    
    #                 return {
    #                     "success": True,
    #                     "video_data": video_base64,
    #                     "content_type": "video/mp4",
    #                     "metrics": {
    #                         "total_tokens": self.token_callback.total_tokens,
    #                         "successful_requests": self.token_callback.successful_requests,
    #                         "failed_requests": self.token_callback.failed_requests,
    #                         "hardcoded_images": True,
    #                         "method": "hardcoded_images_pipeline"
    #                     }
    #                 }
                    
    #             except Exception as e:
    #                 logger.error(f"Error in video generation pipeline: {str(e)}")
    #                 raise
                
    #             finally:
    #                 if video_manager:
    #                     try:
    #                         video_manager.cleanup()
    #                     except Exception as e:
    #                         logger.error(f"Error during video manager cleanup: {str(e)}")
                    
    #                 if s3_handler:
    #                     try:
    #                         s3_handler.cleanup()
    #                     except Exception as e:
    #                         logger.error(f"Error during S3 handler cleanup: {str(e)}")