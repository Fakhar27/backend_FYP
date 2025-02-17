from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
import base64
from langchain_cohere.react_multi_hop.parsing import parse_answer_with_prefixes
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langsmith import Client
from langsmith.run_helpers import traceable, trace
import asyncio
import aiohttp
import os
import logging
from typing import Optional, Dict, Any, List
from .video_manager import VideoManager
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

load_dotenv()

class ContentRequest(BaseModel):
    """Request model for story generation"""
    prompt: str = Field(..., description="User's content prompt")
    genre: str = Field(..., description="Content category/genre")
    iterations: int = Field(default=4, ge=1, le=10)

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
    def __init__(self, colab_url: Optional[str] = None, voice_url: Optional[str] = None):
        self.token_callback = TokenUsageCallback()
        self.client = Client()
        
        self.llm = ChatCohere(
            cohere_api_key=os.getenv("CO_API_KEY"),
            temperature=0.7,
            max_tokens=150,
            callbacks=[self.token_callback]
        )
        
        self.colab_url = colab_url or os.getenv("COLAB_URL")
        self.voice_url = voice_url or os.getenv("COLAB_URL_2")
        self._session = None
        self._session_refs = 0
        
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

    async def get_session(self):
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_refs = 0
        self._session_refs += 1
        return self._session

    async def _release_session(self):
        """Release session reference"""
        if self._session is None:
            return
        self._session_refs -= 1
        if self._session_refs <= 0 and not self._session.closed:
            await self._session.close()
            self._session = None

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
                
                parsed_content = parse_answer_with_prefixes(response.content, self.prefixes)
                
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

    async def generate_image(self, prompt: str) -> Optional[str]:
        """Generate image using Stable Diffusion API with retries"""
        if not self.colab_url:
            logger.error("COLAB_URL not set")
            return None
            
        retries = 3
        for attempt in range(retries):
            try:
                session = await self.get_session()
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
                            await asyncio.sleep(1)  # Wait before retry
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
            finally:
                await self._release_session()
                
        return None
    
    async def generate_voice(self, text: str) -> Optional[str]:
        """Generate voice narration using Bark API"""
        if not self.voice_url:
            logger.error("Voice URL not set")
            return None
            
        retries = 3
        for attempt in range(retries):
            try:
                session = await self.get_session()
                logger.info(f"Sending voice generation request for text: {text}")
                
                async with session.post(
                    f"{self.voice_url}/generate-speech",
                    json={"text": text},
                    timeout=aiohttp.ClientTimeout(total=120)  
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
            finally:
                await self._release_session()
                
        return None


    @traceable(run_type="chain")
    async def generate_content_pipeline(self, request: ContentRequest) -> Dict[str, Any]:
        """Generate complete story with images and voice narration, return as video"""
        with trace(
            name="Full Story Generation",
            run_type="chain",
            project_name=os.getenv("LANGSMITH_PROJECT")
        ) as run:
            video_manager = None
            try:
                video_manager = VideoManager()
                previous_content = None
                
                for i in range(request.iterations):
                    logger.info(f"Starting iteration {i + 1}")
                    
                    try:
                        # Generate story and image description
                        iteration_result = await self.generate_iteration(
                            input_text=request.prompt if i == 0 else "", 
                            genre=request.genre,
                            previous_content=previous_content
                        )
                        
                        # Parallel generation of image and voice
                        image_task = asyncio.create_task(
                            self.generate_image(iteration_result["image"])
                        )
                        voice_task = asyncio.create_task(
                            self.generate_voice(iteration_result["story"])
                        )
                        
                        # Wait for both tasks to complete
                        results = await asyncio.gather(
                            image_task, 
                            voice_task, 
                            return_exceptions=True
                        )
                        
                        # Check for exceptions in gathered results
                        for result in results:
                            if isinstance(result, Exception):
                                raise result
                                
                        image_data, audio_data = results
                        
                        if not image_data or not audio_data:
                            raise ValueError(f"Failed to generate media for iteration {i + 1}")
                        
                        # Create segment data
                        segment_data = {
                            'image_data': image_data,
                            'audio_data': audio_data,
                            'story_text': iteration_result["story"]
                        }
                        
                        # Process segment
                        video_manager.create_segment(segment_data, i)
                        
                        previous_content = iteration_result
                        
                        # Add metadata for tracking
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
                        raise
                
                # Concatenate all segments
                logger.info("Starting video concatenation")
                final_video_path = video_manager.concatenate_segments()
                
                # Read and encode final video
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
    

    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            self._session_refs = 0

    def __del__(self):
        """Ensure cleanup on deletion"""
        if self._session and not self._session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
                else:
                    loop.run_until_complete(self.cleanup())
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                
                
                







               
# @traceable(run_type="chain")
    # async def generate_content_pipeline(self, request: ContentRequest) -> Dict[str, Any]:
    #     """Generate complete story with images and voice narration, return as video"""
    #     with trace(
    #         name="Full Story Generation",
    #         run_type="chain",
    #         project_name=os.getenv("LANGSMITH_PROJECT")
    #     ) as run:
    #         try:
    #             video_manager = VideoManager()
    #             final_video_path = None
                
    #             previous_content = None
                
    #             for i in range(request.iterations):
    #                 logger.info(f"Starting iteration {i + 1}")
                    
    #                 # Generate story and image description
    #                 iteration_result = await self.generate_iteration(
    #                     input_text=request.prompt if i == 0 else "", 
    #                     genre=request.genre,
    #                     previous_content=previous_content
    #                 )
                    
    #                 # Parallel generation of image and voice
    #                 image_task = asyncio.create_task(
    #                     self.generate_image(iteration_result["image"])
    #                 )
    #                 voice_task = asyncio.create_task(
    #                     self.generate_voice(iteration_result["story"])
    #                 )
                    
    #                 # Wait for both tasks to complete
    #                 image_data, audio_data = await asyncio.gather(image_task, voice_task)
                    
    #                 if not image_data or not audio_data:
    #                     raise ValueError(f"Failed to generate media for iteration {i + 1}")
                    
    #                 # Create video segment
    #                 segment = VideoSegment(
    #                     image_data=image_data,
    #                     audio_data=audio_data,
    #                     story_text=iteration_result["story"]
    #                 )
                    
    #                 # Process segment
    #                 video_manager.create_segment(segment, i)
                    
    #                 previous_content = iteration_result
                    
    #                 # Add metadata for tracking
    #                 run.add_metadata({
    #                     f"iteration_{i+1}": {
    #                         "story": iteration_result["story"],
    #                         "image_description": iteration_result["image"],
    #                         "status": "processed",
    #                         "genre": request.genre
    #                     }
    #                 })
                    
    #                 logger.info(f"Completed iteration {i + 1}")
                
    #             final_video_path = video_manager.concatenate_segments()
    #             with open(final_video_path, 'rb') as video_file:
    #                 video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
                
    #             return {
    #                 "success": True,
    #                 "video_data": video_base64,
    #                 "content_type": "video/mp4",
    #                 "metrics": {
    #                     "total_tokens": self.token_callback.total_tokens,
    #                     "successful_requests": self.token_callback.successful_requests,
    #                     "failed_requests": self.token_callback.failed_requests
    #                 }
    #             }
                
    #         except Exception as e:
    #             logger.error(f"Error in video generation pipeline: {str(e)}")
    #             raise
            
    #         finally:
    #             if 'video_manager' in locals():
    #                 video_manager.cleanup()
    
    # @traceable(run_type="chain")
    # async def generate_content_pipeline(self, request: ContentRequest) -> List[ContentResponse]:
    #     """Generate complete story with images and voice narration based on ContentRequest."""
    #     with trace(
    #         name="Full Story Generation",
    #         run_type="chain",
    #         project_name=os.getenv("LANGSMITH_PROJECT")
    #     ) as run:
    #         results = []
    #         previous_content = None
            
    #         for i in range(request.iterations):
    #             logger.info(f"Starting iteration {i + 1}")
                
    #             # Generate story and image description
    #             iteration_result = await self.generate_iteration(
    #                 input_text=request.prompt if i == 0 else "", 
    #                 genre=request.genre,
    #                 previous_content=previous_content
    #             )
                
    #             # Parallel generation of image and voice
    #             image_task = asyncio.create_task(
    #                 self.generate_image(iteration_result["image"])
    #             )
    #             voice_task = asyncio.create_task(
    #                 self.generate_voice(iteration_result["story"])
    #             )
                
    #             # Wait for both tasks to complete
    #             image_url, voice_data = await asyncio.gather(image_task, voice_task)
                
    #             response = ContentResponse(
    #                 story=iteration_result["story"],
    #                 image_description=iteration_result["image"],
    #                 image_url=image_url,
    #                 voice_data=voice_data,  
    #                 iteration=i + 1
    #             )
                
    #             results.append(response)
    #             previous_content = iteration_result
                
    #             run.add_metadata({
    #                 f"iteration_{i+1}": {
    #                     "story": iteration_result["story"],
    #                     "image_description": iteration_result["image"],
    #                     "image_url": image_url,
    #                     "voice_data": "Generated" if voice_data else "Failed",
    #                     "genre": request.genre
    #                 }
    #             })
                
    #             logger.info(f"Completed iteration {i + 1}")
            
    #         return results