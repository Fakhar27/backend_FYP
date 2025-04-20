from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
import base64
from langchain_cohere.react_multi_hop.parsing import parse_answer_with_prefixes
from langchain_core.callbacks import BaseCallbackHandler
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
    useHfInference: bool = Field(default=False, description="Whether to use Hugging Face Inference API")
    hfImageModel: str = Field(default="black-forest-labs/FLUX.1-schnell", description="HF model for image generation")
    useHfVideo: bool = Field(default=False, description="Whether to use Hugging Face for video generation")
    hfVideoModel: str = Field(default="Wan-AI/Wan2.1-T2V-14B", description="HF model for video generation")

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
        self.voice_url = voice_url or os.getenv("COLAB_URL_2")
        self.whisper_url = whisper_url or os.getenv("COLAB_URL_3")
        
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

    async def generate_image_hf(self, prompt: str, model_id: str = "black-forest-labs/FLUX.1-schnell") -> Optional[str]:
        """Generate image using Hugging Face API with direct API calls"""
        try:
            logger.info(f"Using HuggingFace direct API for image generation with model: {model_id}")
            logger.info(f"Prompt: {prompt}")

            # Import here to avoid circular imports (as in your original code)
            # NOTE: This assumes huggingface_service contains the actual implementation
            from .huggingface_service import generate_image_with_hf

            # Call the direct API function (assuming it's correctly implemented in huggingface_service)
            image_data = await generate_image_with_hf(prompt, model_id)

            if not image_data:
                logger.error("No image data returned from Hugging Face")
                return None

            logger.info("Successfully generated image with Hugging Face")
            return image_data

        except Exception as e:
            logger.error(f"Error in HF image generation: {str(e)}")
            # logger.error(traceback.format_exc())
            return None


    async def generate_video_hf(self, prompt: str, model_id: str = "cerspense/zeroscope_v2_XL") -> Optional[str]:
        """Generate video directly using Hugging Face API"""
        try:
            logger.info(f"Using HuggingFace direct API for video generation with model: {model_id}")
            logger.info(f"Prompt: {prompt}")

            # Import here to avoid circular imports (as in your original code)
            # NOTE: This assumes huggingface_service contains the actual implementation
            from .huggingface_service import generate_video_with_hf

            # Call the direct API function (assuming it's correctly implemented in huggingface_service)
            video_data = await generate_video_with_hf(prompt, model_id)

            if not video_data:
                logger.error("No video data returned from Hugging Face")
                return None

            logger.info("Successfully generated video with Hugging Face")
            return video_data

        except Exception as e:
            logger.error(f"Error in HF video generation: {str(e)}")
            # logger.error(traceback.format_exc())
            return None
                            
    
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
                    # Log the request parameters
                    logger.info(f"Processing request with settings: genre={request.genre}, "
                            f"background={request.backgroundVideo}, music={request.backgroundMusic}, "
                            f"voice={request.voiceType}, color={request.subtitleColor}")
                    logger.info(f"Using Hugging Face: {request.useHfInference}")
                    logger.info(f"Hugging Face Video: {request.useHfVideo}")
                    
                    # Verify required services
                    if not self.voice_url:
                        raise ValueError("Voice URL is required")
                        
                    if not self.whisper_url:
                        raise ValueError("Whisper URL is required")
                    
                    # Check available image generation methods
                    use_hf_for_images = False
                    if not self.colab_url:
                        # No Colab URL, must use Hugging Face
                        logger.info("No Colab URL available, will use Hugging Face for image generation")
                        use_hf_for_images = True
                    elif request.useHfInference:
                        # Explicitly requested Hugging Face
                        logger.info("Explicitly requested Hugging Face for image generation")
                        use_hf_for_images = True
                    
                    # Initialize services
                    video_manager = VideoManager()
                    s3_handler = S3Handler()
                    
                    # Direct text-to-video generation if requested
                    if request.useHfVideo:
                        logger.info(f"Attempting direct text-to-video generation with model {request.hfVideoModel}")
                        try:
                            # Import the Hugging Face service only when needed
                            from .huggingface_service import generate_video_with_hf
                            
                            # Generate the video directly with Hugging Face
                            video_data = await generate_video_with_hf(
                                prompt=request.prompt,
                                model_id=request.hfVideoModel
                            )
                            
                            if video_data:
                                # Successfully generated video, return it directly
                                logger.info("Successfully generated video with Hugging Face")
                                
                                # Clean up base64 prefix if present
                                if "base64," in video_data:
                                    video_data = video_data.split("base64,")[1]
                                
                                return {
                                    "success": True,
                                    "video_data": video_data,
                                    "content_type": "video/mp4",
                                    "metrics": {
                                        "total_tokens": self.token_callback.total_tokens,
                                        "successful_requests": self.token_callback.successful_requests,
                                        "failed_requests": self.token_callback.failed_requests,
                                        "method": "huggingface_direct_video"
                                    }
                                }
                            else:
                                logger.warning("Failed to generate video directly, falling back to standard pipeline")
                        except Exception as e:
                            logger.error(f"Error in direct video generation: {str(e)}")
                            logger.exception("Full error details:")
                            logger.info("Falling back to standard pipeline")
                    
                    # If we get here, we're using the standard iterative pipeline
                    previous_content = None
                    segments_data = []
                    
                    # Process each iteration
                    for i in range(request.iterations):
                        try:
                            logger.info(f"\n=== Processing Iteration {i + 1}/{request.iterations} ===")
                            
                            # Generate story content
                            iteration_result = await self.generate_iteration(
                                input_text=request.prompt if i == 0 else "",
                                genre=request.genre,
                                previous_content=previous_content
                            )
                            
                            # Generate image based on available methods
                            if use_hf_for_images:
                                logger.info(f"Using Hugging Face for image generation with model {request.hfImageModel}")
                                
                                # Import the Hugging Face service only when needed
                                from .huggingface_service import generate_image_with_hf
                                
                                image_task = asyncio.create_task(
                                    generate_image_with_hf(
                                        prompt=iteration_result["image"],
                                        model_id=request.hfImageModel
                                    )
                                )
                            else:
                                logger.info(f"Using Colab for image generation at URL: {self.colab_url}")
                                image_task = asyncio.create_task(
                                    self.generate_image(iteration_result["image"], session)
                                )
                            
                            # Generate voice narration
                            voice_task = asyncio.create_task(
                                self.generate_voice(
                                    text=iteration_result["story"], 
                                    voice_type=request.voiceType,
                                    session=session
                                )
                            )
                            
                            # Wait for both tasks to complete
                            logger.info("Waiting for image and voice generation to complete...")
                            image_data, audio_data = await asyncio.gather(
                                image_task,
                                voice_task,
                                return_exceptions=False 
                            )
                            
                            # Validate media generation
                            if not image_data:
                                raise ValueError(f"Failed to generate image for iteration {i + 1}")
                                
                            if not audio_data:
                                raise ValueError(f"Failed to generate audio for iteration {i + 1}")
                            
                            logger.info(f"Successfully generated media for iteration {i + 1}")
                            
                            # Create segment data
                            segment_data = {
                                'image_data': image_data,
                                'audio_data': audio_data,
                                'story_text': iteration_result["story"],
                                'subtitle_color': request.subtitleColor
                            }
                            
                            # Create video segment
                            logger.info(f"Creating video segment for iteration {i + 1}")
                            segment_path = await video_manager.create_segment(
                                segment_data,
                                i,
                                whisper_url=self.whisper_url,
                                session=session
                            )
                            
                            previous_content = iteration_result
                            segments_data.append(segment_path)
                            
                            # Add metadata for tracing
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
                            logger.exception("Full error details:")
                            raise ValueError(f"Failed in iteration {i + 1}: {str(e)}")
                    
                    # Get background video and music files from S3
                    logger.info("Retrieving background media from S3")
                    background_video_path = s3_handler.get_media_file('video', request.backgroundVideo)
                    background_audio_path = s3_handler.get_media_file('music', request.backgroundMusic)
                    
                    logger.info(f"Selected background video: {background_video_path}")
                    logger.info(f"Selected background music: {background_audio_path}")
                    
                    # Handle special "NONE" value or missing files
                    if not background_video_path or background_video_path == "NONE":
                        background_video_path = "NONE" if background_video_path == "NONE" else "E:\\fyp_backend\\backend\\genAI\\split_screen_video_1.mp4"
                        logger.info(f"Using video path: {background_video_path}")
                    
                    if not background_audio_path or background_audio_path == "NONE":
                        background_audio_path = "NONE" if background_audio_path == "NONE" else "E:\\fyp_backend\\backend\\genAI\\backgroundMusic1.wav"
                        logger.info(f"Using audio path: {background_audio_path}")
                    
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
                    
                    return {
                        "success": True,
                        "video_data": video_base64,
                        "content_type": "video/mp4",
                        "metrics": {
                            "total_tokens": self.token_callback.total_tokens,
                            "successful_requests": self.token_callback.successful_requests,
                            "failed_requests": self.token_callback.failed_requests,
                            "method": "iterative_pipeline",
                            "image_source": "huggingface" if use_hf_for_images else "colab"
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Error in video generation pipeline: {str(e)}")
                    logger.exception("Full error details:")
                    raise
                
                finally:
                    # Clean up resources
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
    
    #WORKSSSSSSS
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
                    
    #                 print(f"Using Whisper endpoint: {self.whisper_url}")
                    
    #                 if not self.whisper_url:
    #                     raise ValueError("Whisper URL is required")
                    
    #                 video_manager = VideoManager()
    #                 s3_handler = S3Handler()
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
    #                 if not background_video_path:
    #                     background_video_path = "E:\\fyp_backend\\backend\\genAI\\split_screen_video_1.mp4"
    #                     logger.warning(f"Using fallback video path: {background_video_path}")
                    
    #                 if not background_audio_path:
    #                     background_audio_path = "E:\\fyp_backend\\backend\\genAI\\backgroundMusic1.wav"
    #                     logger.warning(f"Using fallback audio path: {background_audio_path}")
                    
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
                    
    #                 if s3_handler:
    #                     try:
    #                         s3_handler.cleanup()
    #                     except Exception as e:
    #                         logger.error(f"Error during S3 handler cleanup: {str(e)}")
                            
                            
                            

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
    #                             self.generate_voice(iteration_result["story"], session)
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
    #                             'story_text': iteration_result["story"]
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
                    
    #                 logger.info("Starting video concatenation")
    #                 final_video_path = video_manager.concatenate_segments(
    #                     background_audio_path="E:\\fyp_backend\\backend\\genAI\\backgroundMusic1.wav",
    #                     # split_video_path="E:\\fyp_backend\\backend\\genAI\\split_screen_video_1.mp4",
    #                     )
                    
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
    #     """Generate voice narration using Bark API"""
    #     if not self.voice_url:
    #         logger.error("Voice URL not set")
    #         return None
            
    #     retries = 3
    #     for attempt in range(retries):
    #         try:
    #             logger.info(f"Sending voice generation request for text: {text}")
                
    #             async with session.post(
    #                 f"{self.voice_url}/generate_sound",
    #                 json={"text": text},
    #                 timeout=aiohttp.ClientTimeout(total=300)
    #             ) as response:
    #                 response.raise_for_status()
    #                 result = await response.json()
                    
    #                 if 'error' in result:
    #                     logger.error(f"Error from voice generation: {result['error']}")
    #                     if attempt < retries - 1:
    #                         await asyncio.sleep(1)
    #                         continue
    #                     return None
                        
    #                 audio_data = result.get('audio_data')
    #                 if not audio_data:
    #                     logger.error("No audio data in response")
    #                     if attempt < retries - 1:
    #                         await asyncio.sleep(1)
    #                         continue
    #                     return None
                    
    #                 logger.info("Voice generated successfully")
    #                 return audio_data
                    
    #         except Exception as e:
    #             logger.error(f"Voice generation failed (attempt {attempt + 1}/{retries}): {str(e)}")
    #             if attempt < retries - 1:
    #                 await asyncio.sleep(1)
    #                 continue
    #             return None
                
    #     return None
    
    
    
    
    
    
    
    
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
                            