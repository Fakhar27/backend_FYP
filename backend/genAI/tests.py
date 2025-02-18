from django.test import TestCase
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.react_multi_hop.parsing import parse_answer_with_prefixes
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langsmith import Client
import base64
from langsmith.run_helpers import traceable, trace
import asyncio
import aiohttp
import os
import requests
import json
import time
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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
    def __init__(self, colab_url: Optional[str] = None):
        # Initialize callbacks
        self.token_callback = TokenUsageCallback()
        
        # Initialize LangSmith client
        self.client = Client()
        
        # Initialize LLM
        self.llm = ChatCohere(
            cohere_api_key=os.getenv("CO_API_KEY"),
            temperature=0.7,
            max_tokens=150,
            callbacks=[self.token_callback]
        )
        
        # Initialize session handling
        self.colab_url = colab_url or os.getenv("COLAB_URL")
        self._session = None
        self._session_refs = 0
        
        self.prefixes = {
            "story": "story:",
            "image": "image:"
        }
        
        self.base_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are generating very short story segments and image descriptions.
            Format your response exactly as:
            story: [one sentence story]
            image: [detailed visual description]
            
            Requirements:
            - Keep story extremely brief (one sentence)
            - Make image descriptions specific and visual
            - Use exactly the format shown above"""),
            ("human", "{input_prompt}")
        ])
        
        self.continuation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Continue this short story:
            Previous: {previous_story}
            
            Format your response exactly as:
            story: [one sentence continuation]
            image: [detailed visual description]
            
            Requirements:
            - Write only 1 sentence continuing the story
            - Keep image descriptions focused and specific
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
    async def generate_iteration(self, input_text: str, previous_content: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Generate a single story iteration."""
        try:
            with trace(
                name="Story Generation Step",
                run_type="llm",
                project_name=os.getenv("LANGSMITH_PROJECT")
            ) as run:
                if previous_content is None:
                    prompt = self.base_prompt.format_prompt(input_prompt=input_text)
                else:
                    prompt = self.continuation_prompt.format_prompt(
                        previous_story=previous_content["story"]
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
        """Generate image using Stable Diffusion API"""
        if not self.colab_url:
            logger.error("COLAB_URL not set")
            return None
            
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
                    return None
                    
                image_data = result.get('image_data')
                if not image_data:
                    logger.error("No image data in response")
                    return None
                
                logger.info("Image generated successfully")
                return image_data
                
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return None
        finally:
            await self._release_session()

    @traceable(run_type="chain")
    async def generate_full_story(self, initial_prompt: str, iterations: int = 4) -> List[Dict[str, Any]]:
        """Generate complete story with images."""
        with trace(
            name="Full Story Generation",
            run_type="chain",
            project_name=os.getenv("LANGSMITH_PROJECT")
        ) as run:
            results = []
            previous_content = None
            
            for i in range(iterations):
                logger.info(f"Starting iteration {i + 1}")
                
                # Generate story and image description
                iteration_result = await self.generate_iteration(
                    initial_prompt if i == 0 else "", 
                    previous_content
                )
                
                # Generate image using the image description
                image_url = await self.generate_image(iteration_result["image"])
                
                # Combine results
                full_result = {
                    "story": iteration_result["story"],
                    "image_description": iteration_result["image"],
                    "image_url": image_url,
                    "iteration": i + 1
                }
                
                results.append(full_result)
                previous_content = iteration_result
                
                # Add iteration metadata
                run.add_metadata({
                    f"iteration_{i+1}": full_result
                })
                
                logger.info(f"Completed iteration {i + 1}")
            
            return results

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

# For testing
async def main():
    story_chain = StoryIterationChain(colab_url="your-colab-url")
    
    try:
        initial_prompt = "a man inside a jungle"
        story_iterations = await story_chain.generate_full_story(initial_prompt)
        
        for iteration in story_iterations:
            print(f"\nIteration {iteration['iteration']}:")
            print(f"Story: {iteration['story']}")
            print(f"Image Description: {iteration['image_description']}")
            print(f"Image URL: {iteration['image_url']}")
    finally:
        await story_chain.cleanup()
        
        
        
        
        
        
        
        
        
        
        
        
# def test_speech_processing_pipeline(base_url):
#     """
#     Test the complete speech processing pipeline without timeouts
#     """
#     try:
#         # 1. Generate speech from text
#         text_to_speech_url = f"{base_url}/generate-speech"
#         sample_text = "This is a test of the speech processing system."
        
#         logger.info("="*80)
#         logger.info("Step 1: Generating Speech")
#         logger.info("="*80)
#         logger.info(f"Input text: {sample_text}")
        
#         # Generate speech
#         logger.info("Sending speech generation request...")
#         speech_response = requests.post(
#             text_to_speech_url,
#             json={"text": sample_text},
#             headers={"Content-Type": "application/json"}
#         )
        
#         if not speech_response.ok:
#             logger.error(f"Speech generation failed: {speech_response.text}")
#             return
            
#         response_data = speech_response.json()
#         audio_data = response_data.get("audio_data")
        
#         if not audio_data:
#             logger.error("No audio data received")
#             return
        
#         logger.info("✅ Speech generated successfully!")
#         logger.info(f"Audio data size: {len(audio_data)} bytes")
        
#         # 2. Process the generated audio
#         process_audio_url = f"{base_url}/process-audio"
        
#         logger.info("\n" + "="*80)
#         logger.info("Step 2: Processing Audio")
#         logger.info("="*80)
        
#         # Process audio
#         logger.info("Sending audio processing request...")
#         process_response = requests.post(
#             process_audio_url,
#             json={"audio_data": audio_data},
#             headers={"Content-Type": "application/json"}
#         )
        
#         if not process_response.ok:
#             logger.error(f"Audio processing failed: {process_response.text}")
#             return
            
#         results = process_response.json()
        
#         # Display results
#         logger.info("\n" + "="*80)
#         logger.info("Step 3: Results")
#         logger.info("="*80)
        
#         if results.get("error"):
#             logger.error(f"Error from server: {results['error']}")
#             return
        
#         # Language detection
#         if results.get("detected_language"):
#             logger.info(f"\nDetected language: {results['detected_language']}")
#             logger.info(f"Language probability: {results['language_probability']:.2%}")
        
#         # Word-level results
#         if results.get("word_level"):
#             logger.info(f"\nWord-level timestamps ({len(results['word_level'])} words):")
#             logger.info("-"*50)
#             for i, word in enumerate(results['word_level'][:10]):
#                 logger.info(f"{i+1:2d}. {word['word']:<15} [{word['start']:.2f}s -> {word['end']:.2f}s]")
#             if len(results['word_level']) > 10:
#                 logger.info("... (showing first 10 words only)")
        
#         # Line-level results
#         if results.get("line_level"):
#             logger.info(f"\nLine-level timestamps ({len(results['line_level'])} lines):")
#             logger.info("-"*50)
#             for i, line in enumerate(results['line_level']):
#                 logger.info(f"\nLine {i+1}:")
#                 logger.info(f"Text: {line['text']}")
#                 logger.info(f"Time: [{line['start']:.2f}s -> {line['end']:.2f}s]")
#                 logger.info(f"Words in line: {len(line['words'])}")
        
#         # Save results
#         with open('complete_results.json', 'w') as f:
#             json.dump(results, f, indent=2)
#         logger.info("\n✅ Complete results saved to: complete_results.json")
        
#     except Exception as e:
#         logger.error(f"Test failed: {str(e)}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1:
#         base_url = sys.argv[1]
#     else:
#         base_url = "https://ef76-34-126-71-138.ngrok-free.app"  # Default URL
    
#     logger.info(f"Testing API at: {base_url}")
#     logger.info("Waiting for server to be ready...")
#     time.sleep(2)
    
#     test_speech_processing_pipeline(base_url)












# def test_whisper_api(base_url,wav_file_path):
#     """Test the Whisper transcription API"""
#     try:
#         logger.info("Testing Whisper API")
#         logger.info(f"Base URL: {base_url}")
#         logger.info(f"WAV file: {wav_file_path}")
        
#         # Prepare the file for upload
#         with open(wav_file_path, 'rb') as f:
#             files = {'file': (wav_file_path, f, 'audio/wav')}
            
#             # Send request
#             logger.info("Sending request...")
#             response = requests.post(
#                 f"{base_url}/process-audio",
#                 files=files
#             )
        
#         if not response.ok:
#             logger.error(f"Request failed: {response.text}")
#             return
            
#         # Process results
#         results = response.json()
        
#         # Save complete results
#         with open('api_results.json', 'w') as f:
#             json.dump(results, f, indent=2)
#         logger.info("Results saved to api_results.json")
        
#         # Display sample of results
#         if results.get("word_level"):
#             logger.info("\nFirst 5 words:")
#             for word in results["word_level"][:5]:
#                 logger.info(f"Word: {word['word']:<15} [{word['start']:.2f}s -> {word['end']:.2f}s]")
        
#         if results.get("line_level"):
#             logger.info("\nFirst 2 lines:")
#             for line in results["line_level"][:2]:
#                 logger.info(f"\nLine: {line['text']}")
#                 logger.info(f"Time: [{line['start']:.2f}s -> {line['end']:.2f}s]")
        
#     except Exception as e:
#         logger.error(f"Test failed: {str(e)}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     # import sys
    
#     # if len(sys.argv) != 3:
#     #     print("Usage: python test_script.py [base_url] [wav_file_path]")
#     #     sys.exit(1)
        
#     # base_url = sys.argv[1]
#     # wav_file_path = sys.argv[2]
    
#     test_whisper_api("https://39f0-34-126-71-138.ngrok-free.app","./temp_audio_1305.wav")









def test_whisper_api(base_url, wav_file_path):
    """Test the Whisper transcription API with base64 encoding"""
    try:
        logger.info("Testing Whisper API")
        logger.info(f"Base URL: {base_url}")
        logger.info(f"WAV file: {wav_file_path}")
        
        # Read and encode the WAV file
        logger.info("Reading and encoding WAV file...")
        with open(wav_file_path, 'rb') as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        logger.info(f"File encoded (size: {len(audio_base64)} bytes)")
        
        # Send request
        logger.info("Sending request...")
        response = requests.post(
            f"{base_url}/audio-process",
            json={"audio_data": audio_base64,
                  "type": "transcribe"},
            headers={"Content-Type": "application/json"}
        )
        
        if not response.ok:
            logger.error(f"Request failed: {response.text}")
            return
            
        # Process results
        results = response.json()
        
        # Save complete results
        with open('api_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to api_results.json")
        
        # Display sample of results
        if results.get("word_level"):
            logger.info("\nFirst 5 words:")
            for word in results["word_level"][:5]:
                logger.info(f"Word: {word['word']:<15} [{word['start']:.2f}s -> {word['end']:.2f}s]")
        
        if results.get("line_level"):
            logger.info("\nFirst 2 lines:")
            for line in results["line_level"][:2]:
                logger.info(f"\nLine: {line['text']}")
                logger.info(f"Time: [{line['start']:.2f}s -> {line['end']:.2f}s]")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # import sys
    
    # if len(sys.argv) != 3:
    #     print("Usage: python test_script.py [base_url] [wav_file_path]")
    #     sys.exit(1)
        
    # base_url = sys.argv[1]
    # wav_file_path = sys.argv[2]
    
    test_whisper_api("https://9aaa-34-19-39-114.ngrok-free.app", "./temp_audio_1305.wav")