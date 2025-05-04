import asyncio
import json
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class ContentRequest:
    def __init__(self, prompt, genre, iterations, negative_prompt, guidance_scale):
        self.prompt = prompt
        self.genre = genre
        self.iterations = iterations
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale

class StoryIterationChain:  # Replace with your actual class and method
    @staticmethod
    async def generate_video_WAN(content_request):
        # Simulate your video generation logic here
        await asyncio.sleep(1)  # Simulate some async work
        return {
            "video_data": "fake_video_data_base64",
            "content_type": "video/mp4",
            "metrics": {"generation_time": 1.5}
        }

@csrf_exempt
async def wan_video_generation_request(request):
    try:
        data = json.loads(request.body.decode('utf-8'))  # Access request body as bytes and decode
        prompt = data.get("prompt")
        genre = "cinematic"
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        iterations = 3
        guidance_scale = 5

        content_request = ContentRequest(
            prompt=prompt,
            genre=genre,
            iterations=iterations,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale
        )

        logger.info(f"Content request: {content_request.__dict__}")

        result = await StoryIterationChain.generate_video_WAN(content_request)

        response_data = {
            "success": True,
            "video_data": result["video_data"],
            "content_type": result["content_type"],
            "metrics": result["metrics"]
        }

        logger.info("Returning video response")
        return JsonResponse(response_data, status=200)
    except Exception as e:
        error_msg = f"Content generation error: {str(e)}"
        logger.error(error_msg)
        return JsonResponse({"error": error_msg}, status=500)

async def main():
    # Simulate an incoming POST request with JSON data
    mock_request = HttpRequest()
    mock_request.method = 'POST'
    mock_request._body = json.dumps({"prompt": "boat on a serene lake at sunset"}).encode('utf-8')
    mock_request.META['CONTENT_TYPE'] = 'application/json'  # Set the content type header

    response = await wan_video_generation_request(mock_request)

    # Process the response
    if isinstance(response, JsonResponse):
        content = json.loads(response.content.decode('utf-8'))
        print("Response:", content)
        print("Status Code:", response.status_code)
    else:
        print("Unexpected response:", response)

if __name__ == "__main__":
    asyncio.run(main())

# from django.test import TestCase
# from langchain_cohere import ChatCohere
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_cohere.react_multi_hop.parsing import parse_answer_with_prefixes
# from langchain_core.callbacks import BaseCallbackHandler
# from langchain_core.outputs import LLMResult
# from langsmith import Client
# import base64
# from langsmith.run_helpers import traceable, trace
# import asyncio
# import aiohttp
# import os
# from .views import wan_video_generation_request
# import requests
# import json
# import time
# import logging
# from typing import Optional, Dict, Any, List
# from dotenv import load_dotenv

# # Configure logging
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# class TokenUsageCallback(BaseCallbackHandler):
#     """Callback handler to track token usage."""
#     def __init__(self):
#         super().__init__()
#         self.total_tokens = 0
#         self.prompt_tokens = 0
#         self.completion_tokens = 0
#         self.successful_requests = 0
#         self.failed_requests = 0

#     def on_llm_start(self, *args, **kwargs) -> None:
#         """Called when LLM starts processing."""
#         pass

#     def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
#         """Called when LLM ends processing."""
#         if response.llm_output and "token_usage" in response.llm_output:
#             usage = response.llm_output["token_usage"]
#             self.total_tokens += usage.get("total_tokens", 0)
#             self.prompt_tokens += usage.get("prompt_tokens", 0)
#             self.completion_tokens += usage.get("completion_tokens", 0)
#             self.successful_requests += 1
#             logger.info(f"Token usage updated - Total: {self.total_tokens}")

#     def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
#         """Called when LLM errors during processing."""
#         self.failed_requests += 1
#         logger.error(f"LLM error occurred: {str(error)}")

# class StoryIterationChain:
#     def __init__(self, colab_url: Optional[str] = None):
#         # Initialize callbacks
#         self.token_callback = TokenUsageCallback()
        
#         # Initialize LangSmith client
#         self.client = Client()
        
#         # Initialize LLM
#         self.llm = ChatCohere(
#             cohere_api_key=os.getenv("CO_API_KEY"),
#             temperature=0.7,
#             max_tokens=150,
#             callbacks=[self.token_callback]
#         )
        
#         # Initialize session handling
#         self.colab_url = colab_url or os.getenv("COLAB_URL")
#         self._session = None
#         self._session_refs = 0
        
#         self.prefixes = {
#             "story": "story:",
#             "image": "image:"
#         }
        
#         self.base_prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are generating very short story segments and image descriptions.
#             Format your response exactly as:
#             story: [one sentence story]
#             image: [detailed visual description]
            
#             Requirements:
#             - Keep story extremely brief (one sentence)
#             - Make image descriptions specific and visual
#             - Use exactly the format shown above"""),
#             ("human", "{input_prompt}")
#         ])
        
#         self.continuation_prompt = ChatPromptTemplate.from_messages([
#             ("system", """Continue this short story:
#             Previous: {previous_story}
            
#             Format your response exactly as:
#             story: [one sentence continuation]
#             image: [detailed visual description]
            
#             Requirements:
#             - Write only 1 sentence continuing the story
#             - Keep image descriptions focused and specific
#             - Use exactly the format shown above"""),
#             ("human", "Continue the story.")
#         ])

#     async def get_session(self):
#         """Get or create aiohttp session"""
#         if self._session is None or self._session.closed:
#             self._session = aiohttp.ClientSession()
#             self._session_refs = 0
#         self._session_refs += 1
#         return self._session

#     async def _release_session(self):
#         """Release session reference"""
#         if self._session is None:
#             return
#         self._session_refs -= 1
#         if self._session_refs <= 0 and not self._session.closed:
#             await self._session.close()
#             self._session = None

#     @traceable(run_type="chain")
#     async def generate_iteration(self, input_text: str, previous_content: Optional[Dict[str, str]] = None) -> Dict[str, str]:
#         """Generate a single story iteration."""
#         try:
#             with trace(
#                 name="Story Generation Step",
#                 run_type="llm",
#                 project_name=os.getenv("LANGSMITH_PROJECT")
#             ) as run:
#                 if previous_content is None:
#                     prompt = self.base_prompt.format_prompt(input_prompt=input_text)
#                 else:
#                     prompt = self.continuation_prompt.format_prompt(
#                         previous_story=previous_content["story"]
#                     )
                
#                 response = await self.llm.ainvoke(
#                     prompt.to_messages()
#                 )
                
#                 parsed_content = parse_answer_with_prefixes(response.content, self.prefixes)
                
#                 # Add run metadata
#                 run.add_metadata({
#                     "token_usage": {
#                         "total_tokens": self.token_callback.total_tokens,
#                         "prompt_tokens": self.token_callback.prompt_tokens,
#                         "completion_tokens": self.token_callback.completion_tokens
#                     },
#                     "request_stats": {
#                         "successful": self.token_callback.successful_requests,
#                         "failed": self.token_callback.failed_requests
#                     }
#                 })
                
#                 return parsed_content
                
#         except Exception as e:
#             logger.error(f"Error in generation: {str(e)}")
#             return {
#                 "story": "Error occurred in story generation.",
#                 "image": "Error occurred in image description."
#             }

#     async def generate_image(self, prompt: str) -> Optional[str]:
#         """Generate image using Stable Diffusion API"""
#         if not self.colab_url:
#             logger.error("COLAB_URL not set")
#             return None
            
#         try:
#             session = await self.get_session()
#             logger.info(f"Sending image generation request with prompt: {prompt}")
            
#             async with session.post(
#                 f"{self.colab_url}/generate-image",
#                 json={"prompt": prompt},
#                 timeout=aiohttp.ClientTimeout(total=60)
#             ) as response:
#                 response.raise_for_status()
#                 result = await response.json()
                
#                 if 'error' in result:
#                     logger.error(f"Error from image generation: {result['error']}")
#                     return None
                    
#                 image_data = result.get('image_data')
#                 if not image_data:
#                     logger.error("No image data in response")
#                     return None
                
#                 logger.info("Image generated successfully")
#                 return image_data
                
#         except Exception as e:
#             logger.error(f"Image generation failed: {str(e)}")
#             return None
#         finally:
#             await self._release_session()

#     @traceable(run_type="chain")
#     async def generate_full_story(self, initial_prompt: str, iterations: int = 4) -> List[Dict[str, Any]]:
#         """Generate complete story with images."""
#         with trace(
#             name="Full Story Generation",
#             run_type="chain",
#             project_name=os.getenv("LANGSMITH_PROJECT")
#         ) as run:
#             results = []
#             previous_content = None
            
#             for i in range(iterations):
#                 logger.info(f"Starting iteration {i + 1}")
                
#                 # Generate story and image description
#                 iteration_result = await self.generate_iteration(
#                     initial_prompt if i == 0 else "", 
#                     previous_content
#                 )
                
#                 # Generate image using the image description
#                 image_url = await self.generate_image(iteration_result["image"])
                
#                 # Combine results
#                 full_result = {
#                     "story": iteration_result["story"],
#                     "image_description": iteration_result["image"],
#                     "image_url": image_url,
#                     "iteration": i + 1
#                 }
                
#                 results.append(full_result)
#                 previous_content = iteration_result
                
#                 # Add iteration metadata
#                 run.add_metadata({
#                     f"iteration_{i+1}": full_result
#                 })
                
#                 logger.info(f"Completed iteration {i + 1}")
            
#             return results

#     async def cleanup(self):
#         """Cleanup resources"""
#         if self._session and not self._session.closed:
#             await self._session.close()
#             self._session = None
#             self._session_refs = 0

#     def __del__(self):
#         """Ensure cleanup on deletion"""
#         if self._session and not self._session.closed:
#             try:
#                 loop = asyncio.get_event_loop()
#                 if loop.is_running():
#                     loop.create_task(self.cleanup())
#                 else:
#                     loop.run_until_complete(self.cleanup())
#             except Exception as e:
#                 logger.error(f"Error during cleanup: {str(e)}")

# # For testing
# async def main():
#     prompt = "boat on a lake"
#     await wan_video_generation_request()
    
        
        
        
        
        
        
        
        
        
        
        
        
# # def test_speech_processing_pipeline(base_url):
# #     """
# #     Test the complete speech processing pipeline without timeouts
# #     """
# #     try:
# #         # 1. Generate speech from text
# #         text_to_speech_url = f"{base_url}/generate-speech"
# #         sample_text = "This is a test of the speech processing system."
        
# #         logger.info("="*80)
# #         logger.info("Step 1: Generating Speech")
# #         logger.info("="*80)
# #         logger.info(f"Input text: {sample_text}")
        
# #         # Generate speech
# #         logger.info("Sending speech generation request...")
# #         speech_response = requests.post(
# #             text_to_speech_url,
# #             json={"text": sample_text},
# #             headers={"Content-Type": "application/json"}
# #         )
        
# #         if not speech_response.ok:
# #             logger.error(f"Speech generation failed: {speech_response.text}")
# #             return
            
# #         response_data = speech_response.json()
# #         audio_data = response_data.get("audio_data")
        
# #         if not audio_data:
# #             logger.error("No audio data received")
# #             return
        
# #         logger.info("✅ Speech generated successfully!")
# #         logger.info(f"Audio data size: {len(audio_data)} bytes")
        
# #         # 2. Process the generated audio
# #         process_audio_url = f"{base_url}/process-audio"
        
# #         logger.info("\n" + "="*80)
# #         logger.info("Step 2: Processing Audio")
# #         logger.info("="*80)
        
# #         # Process audio
# #         logger.info("Sending audio processing request...")
# #         process_response = requests.post(
# #             process_audio_url,
# #             json={"audio_data": audio_data},
# #             headers={"Content-Type": "application/json"}
# #         )
        
# #         if not process_response.ok:
# #             logger.error(f"Audio processing failed: {process_response.text}")
# #             return
            
# #         results = process_response.json()
        
# #         # Display results
# #         logger.info("\n" + "="*80)
# #         logger.info("Step 3: Results")
# #         logger.info("="*80)
        
# #         if results.get("error"):
# #             logger.error(f"Error from server: {results['error']}")
# #             return
        
# #         # Language detection
# #         if results.get("detected_language"):
# #             logger.info(f"\nDetected language: {results['detected_language']}")
# #             logger.info(f"Language probability: {results['language_probability']:.2%}")
        
# #         # Word-level results
# #         if results.get("word_level"):
# #             logger.info(f"\nWord-level timestamps ({len(results['word_level'])} words):")
# #             logger.info("-"*50)
# #             for i, word in enumerate(results['word_level'][:10]):
# #                 logger.info(f"{i+1:2d}. {word['word']:<15} [{word['start']:.2f}s -> {word['end']:.2f}s]")
# #             if len(results['word_level']) > 10:
# #                 logger.info("... (showing first 10 words only)")
        
# #         # Line-level results
# #         if results.get("line_level"):
# #             logger.info(f"\nLine-level timestamps ({len(results['line_level'])} lines):")
# #             logger.info("-"*50)
# #             for i, line in enumerate(results['line_level']):
# #                 logger.info(f"\nLine {i+1}:")
# #                 logger.info(f"Text: {line['text']}")
# #                 logger.info(f"Time: [{line['start']:.2f}s -> {line['end']:.2f}s]")
# #                 logger.info(f"Words in line: {len(line['words'])}")
        
# #         # Save results
# #         with open('complete_results.json', 'w') as f:
# #             json.dump(results, f, indent=2)
# #         logger.info("\n✅ Complete results saved to: complete_results.json")
        
# #     except Exception as e:
# #         logger.error(f"Test failed: {str(e)}")
# #         import traceback
# #         traceback.print_exc()

# # if __name__ == "__main__":
# #     import sys
    
# #     if len(sys.argv) > 1:
# #         base_url = sys.argv[1]
# #     else:
# #         base_url = "https://ef76-34-126-71-138.ngrok-free.app"  # Default URL
    
# #     logger.info(f"Testing API at: {base_url}")
# #     logger.info("Waiting for server to be ready...")
# #     time.sleep(2)
    
# #     test_speech_processing_pipeline(base_url)












# # def test_whisper_api(base_url,wav_file_path):
# #     """Test the Whisper transcription API"""
# #     try:
# #         logger.info("Testing Whisper API")
# #         logger.info(f"Base URL: {base_url}")
# #         logger.info(f"WAV file: {wav_file_path}")
        
# #         # Prepare the file for upload
# #         with open(wav_file_path, 'rb') as f:
# #             files = {'file': (wav_file_path, f, 'audio/wav')}
            
# #             # Send request
# #             logger.info("Sending request...")
# #             response = requests.post(
# #                 f"{base_url}/process-audio",
# #                 files=files
# #             )
        
# #         if not response.ok:
# #             logger.error(f"Request failed: {response.text}")
# #             return
            
# #         # Process results
# #         results = response.json()
        
# #         # Save complete results
# #         with open('api_results.json', 'w') as f:
# #             json.dump(results, f, indent=2)
# #         logger.info("Results saved to api_results.json")
        
# #         # Display sample of results
# #         if results.get("word_level"):
# #             logger.info("\nFirst 5 words:")
# #             for word in results["word_level"][:5]:
# #                 logger.info(f"Word: {word['word']:<15} [{word['start']:.2f}s -> {word['end']:.2f}s]")
        
# #         if results.get("line_level"):
# #             logger.info("\nFirst 2 lines:")
# #             for line in results["line_level"][:2]:
# #                 logger.info(f"\nLine: {line['text']}")
# #                 logger.info(f"Time: [{line['start']:.2f}s -> {line['end']:.2f}s]")
        
# #     except Exception as e:
# #         logger.error(f"Test failed: {str(e)}")
# #         import traceback
# #         traceback.print_exc()

# # if __name__ == "__main__":
# #     # import sys
    
# #     # if len(sys.argv) != 3:
# #     #     print("Usage: python test_script.py [base_url] [wav_file_path]")
# #     #     sys.exit(1)
        
# #     # base_url = sys.argv[1]
# #     # wav_file_path = sys.argv[2]
    
# #     test_whisper_api("https://39f0-34-126-71-138.ngrok-free.app","./temp_audio_1305.wav")



# def find_all_wav_files(start_path='.'):
#     wav_files = []
#     for root, dirs, files in os.walk(start_path):
#         for file in files:
#             if file.endswith('.wav'):
#                 wav_files.append(os.path.join(root, file))
#     return wav_files

# # Print the current working directory
# print(f"Current working directory: {os.getcwd()}")

# # Find all .wav files
# wav_files = find_all_wav_files()
# print(f"Found {len(wav_files)} .wav files:") 
# for file in wav_files:
#     print(f"  - {file} (exists: {os.path.exists(file)})")


# # def test_audio_generation(base_url):
# #     """Simple test for the audio generation API endpoint"""
# #     try:
# #         logger.info("Testing Audio Generation API")
# #         logger.info(f"Base URL: {base_url}")
        
# #         # Simple test text
# #         test_text = "This is a test for audio generation."
        
# #         # Send request
# #         response = requests.post(
# #             f"{base_url}/generate_sound",
# #             json={
# #                 "text": test_text,
# #             },
# #             headers={"Content-Type": "application/json"},
# #             timeout=500
# #         )
        
# #         if response.ok:
# #             print("WORKED SOUND GENERATION!!!!")
# #             logger.info("✓ API test successful - got valid response")
# #             logger.info(f"Status code: {response.status_code}")
# #         else:
# #             logger.error(f"✗ API test failed")
# #             logger.error(f"Status: {response.status_code}")
# #             logger.error(f"Response: {response.text}")
            
# #     except Exception as e:
# #         logger.error(f"Test failed: {str(e)}")
# #         import traceback
# #         traceback.print_exc()


# # def test_whisper_api(base_url, wav_file_path):
# #     """Test the Whisper transcription API with base64 encoding"""
# #     try:
# #         logger.info("Testing Whisper API")
# #         logger.info(f"Base URL: {base_url}")
# #         logger.info(f"WAV file: {wav_file_path}")
        
# #         # Read and encode the WAV file
# #         logger.info("Reading and encoding WAV file...")
# #         with open(wav_file_path, 'rb') as f:
# #             audio_data = f.read()
# #             audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
# #         logger.info(f"File encoded (size: {len(audio_base64)} bytes)")
        
# #         # Send request
# #         logger.info("Sending request...")
# #         response = requests.post(
# #             f"{base_url}/process_audio",
# #             json={"audio_data": audio_base64,},
# #             headers={"Content-Type": "application/json"}
# #         )
        
# #         if not response.ok:
# #             logger.error(f"Request failed: {response.text}")
# #             return
            
# #         # Process results
# #         results = response.json()
        
# #         # Save complete results
# #         with open('api_results.json', 'w') as f:
# #             json.dump(results, f, indent=2)
# #         logger.info("Results saved to api_results.json")
        
# #         # Display sample of results
# #         if results.get("word_level"):
# #             logger.info("\nFirst 5 words:")
# #             for word in results["word_level"][:5]:
# #                 logger.info(f"Word: {word['word']:<15} [{word['start']:.2f}s -> {word['end']:.2f}s]")
        
# #         if results.get("line_level"):
# #             logger.info("\nFirst 2 lines:")
# #             for line in results["line_level"][:2]:
# #                 logger.info(f"\nLine: {line['text']}")
# #                 logger.info(f"Time: [{line['start']:.2f}s -> {line['end']:.2f}s]")
        
# #     except Exception as e:
# #         logger.error(f"Test failed: {str(e)}")
# #         import traceback
# #         traceback.print_exc()

# # if __name__ == "__main__":
# #     # import sys
    
# #     # if len(sys.argv) != 3:
# #     #     print("Usage: python test_script.py [base_url] [wav_file_path]")
# #     #     sys.exit(1)
        
# #     # base_url = sys.argv[1]
# #     # wav_file_path = sys.argv[2]
    
# #     # test_audio_generation("https://73d3-35-204-183-178.ngrok-free.app")
# #     test_whisper_api("https://ea62-34-16-212-86.ngrok-free.app", "./temp_audio_1305.wav")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# # import wave
# # import pyaudio
# # import os
# # import time

# # def play_wav_file(file_path):
# #     """Play a WAV file"""
# #     try:
# #         # Open the WAV file
# #         wf = wave.open(file_path, 'rb')
        
# #         # Initialize PyAudio
# #         p = pyaudio.PyAudio()
        
# #         # Open stream
# #         stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
# #                        channels=wf.getnchannels(),
# #                        rate=wf.getframerate(),
# #                        output=True)
        
# #         # Read data in chunks and play
# #         chunk_size = 1024
# #         data = wf.readframes(chunk_size)
        
# #         print(f"\nPlaying: {file_path}")
# #         print("Press Ctrl+C to stop playback")
        
# #         while data:
# #             stream.write(data)
# #             data = wf.readframes(chunk_size)
            
# #         # Cleanup
# #         stream.stop_stream()
# #         stream.close()
# #         p.terminate()
# #         wf.close()
        
# #     except Exception as e:
# #         print(f"Error playing audio: {str(e)}")

# # def __main__():
# #     """Test function for audio playback"""
# #     print("RUNNING TESTCASE FUNCTION ONLY")
    
# #     # Define the path to your WAV file
# #     path_of_file = "./backgroundMusic1.wav"
    
# #     # Check if file exists
# #     if os.path.exists(path_of_file):
# #         print(f"YES {path_of_file} exists")
# #         # Play the WAV file
# #         play_wav_file(path_of_file)
# #     else:
# #         print(f"File not found: {path_of_file}")
# #         # Try looking in parent directory
# #         parent_path = "../backgroundMusic1.wav"
# #         if os.path.exists(parent_path):
# #             print(f"Found file in parent directory: {parent_path}")
# #             play_wav_file(parent_path)
# #         else:
# #             print("Could not find backgroundMusic1.wav in current or parent directory")

# # if __name__ == "__main__":
# #     __main__()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# #     import os
# # import base64
# # import tempfile
# # import logging
# # from typing import List, Optional, Dict
# # import numpy as np
# # from PIL import Image
# # import io
# # from datetime import timedelta
# # import json
# # import asyncio
# # import aiohttp
# # from moviepy import *
# # from moviepy.video.tools.subtitles import SubtitlesClip
# # from contextlib import contextmanager

# # logger = logging.getLogger(__name__)

# # class VideoProcessingError(Exception):
# #     """Custom exception for video processing errors"""
# #     pass

# # class VideoManager:
# #     def __init__(self):
# #         """Initialize video manager with temporary directory and font settings"""
# #         self.temp_dir = tempfile.mkdtemp()
# #         self.segments: List[str] = []
# #         self.font_path = self._get_system_font()
# #         logger.info(f"VideoManager initialized with temp dir: {self.temp_dir}")

# #     def _get_system_font(self) -> str:
# #         """Get system font path based on OS"""
# #         font_paths = {
# #             'nt': [  # Windows
# #                 r"C:\Windows\Fonts\Arial.ttf",
# #                 r"C:\Windows\Fonts\Calibri.ttf",
# #                 r"C:\Windows\Fonts\segoeui.ttf"
# #             ],
# #             'posix': [  # Linux/Unix
# #                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
# #                 "/usr/share/fonts/TTF/Arial.ttf",
# #                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
# #             ]
# #         }

# #         paths = font_paths.get(os.name, [])
# #         for path in paths:
# #             if os.path.exists(path):
# #                 logger.info(f"Using system font: {path}")
# #                 return path

# #         logger.warning("No system fonts found, using default")
# #         return ""

# #     def _decode_base64_image(self, base64_str: str) -> np.ndarray:
# #         """Convert base64 image to numpy array"""
# #         try:
# #             base64_str = base64_str.split('base64,')[-1]
# #             image_data = base64.b64decode(base64_str)
# #             image = Image.open(io.BytesIO(image_data))
# #             return np.array(image.convert('RGB'))
# #         except Exception as e:
# #             raise VideoProcessingError(f"Failed to decode image: {e}")

# #     def _save_base64_audio(self, base64_str: str, index: int) -> str:
# #         """Save base64 audio to temporary WAV file"""
# #         try:
# #             base64_str = base64_str.split('base64,')[-1]
# #             audio_data = base64.b64decode(base64_str)
# #             audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
# #             with open(audio_path, 'wb') as f:
# #                 f.write(audio_data)
# #             return audio_path
# #         except Exception as e:
# #             raise VideoProcessingError(f"Failed to save audio: {e}")

    
    
# #     def _create_srt_content(self, text: str, start: float, duration: float) -> str:
# #         def format_time(seconds: float) -> str:
# #             td = timedelta(seconds=max(0, seconds))
# #             hours = td.seconds // 3600
# #             minutes = (td.seconds % 3600) // 60
# #             seconds = td.seconds % 60
# #             ms = round(td.microseconds / 1000)
# #             return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

# #         # Improve text chunking
# #         words = text.split()
# #         chunks = []
# #         current_chunk = []
        
# #         for word in words:
# #             current_chunk.append(word)
# #             # Adjust max chars and consider word boundaries
# #             if len(' '.join(current_chunk)) > 35:  # Shorter lines for better readability
# #                 chunks.append(' '.join(current_chunk[:-1]))
# #                 current_chunk = [word]
# #         if current_chunk:
# #             chunks.append(' '.join(current_chunk))

# #         # Create more precise timing
# #         srt_parts = []
# #         total_words = sum(len(chunk.split()) for chunk in chunks)
# #         words_per_second = total_words / duration
        
# #         current_time = start
# #         for i, chunk in enumerate(chunks, 1):
# #             # Calculate duration based on word count
# #             chunk_words = len(chunk.split())
# #             chunk_duration = (chunk_words / words_per_second) * 1.2  # Add 20% buffer
            
# #             chunk_end = min(current_time + chunk_duration, start + duration)
            
# #             srt_parts.append(
# #                 f"{i}\n"
# #                 f"{format_time(current_time)} --> {format_time(chunk_end)}\n"
# #                 f"{chunk}\n"
# #             )
# #             current_time = chunk_end

# #         return "\n".join(srt_parts)
    
# #     def _format_time(self, seconds: float) -> str:
# #         """Format seconds into SRT timestamp format"""
# #         hours = int(seconds // 3600)
# #         minutes = int((seconds % 3600) // 60)
# #         seconds = seconds % 60
# #         milliseconds = int((seconds - int(seconds)) * 1000)
# #         return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    
# #     async def get_synchronized_subtitles(self, audio_data: str, whisper_url: str, session: aiohttp.ClientSession) -> Dict:
# #         """Get synchronized subtitles for audio using Whisper API"""
# #         try:
# #             logger.info(f"Starting subtitle request to Whisper API at URL: {whisper_url}")
# #             print(f"Attempting to call Whisper API at: {whisper_url}")
            
# #             # Log audio data length for debugging
# #             print(f"Audio data length before processing: {len(audio_data)}")
            
# #             # Ensure audio_data is properly formatted
# #             if ',' in audio_data:
# #                 audio_data = audio_data.split('base64,')[1]
# #                 print(f"Audio data length after base64 split: {len(audio_data)}")
            
# #             logger.info("Preparing API request...")
# #             print("Preparing to send request to Whisper API...")
            
# #             # Use the provided session
# #             try:
# #                 async with session.post(
# #                     f"{whisper_url}/process_audio",
# #                     json={
# #                         "audio_data": audio_data,
# #                     },
# #                     headers={"Content-Type": "application/json"},
# #                     timeout=aiohttp.ClientTimeout(total=500)
# #                 ) as response:
# #                     logger.info(f"Received response from Whisper API. Status: {response.status}")
# #                     print(f"Whisper API Response Status: {response.status}")
                    
# #                     if response.status != 200:
# #                         error_text = await response.text()
# #                         logger.error(f"Whisper API error response: {error_text}")
# #                         print(f"Error from Whisper API: {error_text}")
# #                         raise VideoProcessingError(f"Whisper API error: {error_text}")
                    
# #                     logger.info("Successfully got response, parsing JSON...")
# #                     print("Parsing Whisper API response...")
                    
# #                     transcription_data = await response.json()
# #                     print("\n=== Whisper API Response Data ===")
# #                     print(json.dumps(transcription_data, indent=2))
# #                     print("===============================\n")
                    
# #                     if not transcription_data:
# #                         logger.error("Received empty transcription data")
# #                         print("Warning: Empty transcription data received")
# #                         raise VideoProcessingError("Empty transcription data received")
                    
# #                     if 'line_level' not in transcription_data:
# #                         logger.error(f"Missing line_level in response. Keys received: {transcription_data.keys()}")
# #                         print(f"Missing required data. Keys in response: {transcription_data.keys()}")
# #                         raise VideoProcessingError("Invalid transcription data: missing line_level")
                    
# #                     logger.info(f"Successfully processed transcription data with {len(transcription_data['line_level'])} lines")
# #                     print(f"Found {len(transcription_data['line_level'])} lines of transcription")
                    
# #                     return transcription_data
                        
# #             except aiohttp.ClientError as e:
# #                 logger.error(f"Network error during API call: {str(e)}")
# #                 print(f"Network error occurred: {str(e)}")
# #                 raise VideoProcessingError(f"Network error: {str(e)}")
                
# #         except Exception as e:
# #             logger.error(f"Unexpected error in get_synchronized_subtitles: {str(e)}")
# #             print(f"Error getting subtitles: {str(e)}")
# #             raise VideoProcessingError(f"Failed to get synchronized subtitles: {str(e)}")

# #     def create_word_level_subtitles(self, whisper_data: Dict, frame_size: tuple, duration: float) -> List:
# #         """Creates word-level synchronized subtitle clips"""
# #         try:
# #             print("Starting subtitle clip creation...")
            
# #             def create_word_clip(word_data, is_highlight=False):
# #                 try:
# #                     text = word_data['word'].strip()
# #                     if not text:  # Skip empty words
# #                         print(f"Skipping empty word: {word_data}")
# #                         return None
                        
# #                     clip = TextClip(
# #                         text=text,
# #                         font=self.font_path,
# #                         font_size=int(frame_size[1] * 0.075),
# #                         color='yellow' if is_highlight else 'white',
# #                         stroke_color='black',
# #                         stroke_width=2
# #                     ).with_duration(duration)
                    
# #                     print(f"Created clip for word: {text}")
# #                     return clip
# #                 except Exception as e:
# #                     print(f"Error creating clip for word {word_data}: {str(e)}")
# #                     return None

# #             word_clips = []
# #             x_pos = frame_size[0] * 0.1
# #             y_pos = frame_size[1] * 0.8
# #             line_width = 0
# #             max_width = frame_size[0] * 0.8

# #             # Process each line from whisper data
# #             for line in whisper_data['line_level']:
# #                 print(f"\nProcessing line: {line['text']}")
# #                 current_line_clips = []
                
# #                 for word_data in line['words']:
# #                     # Create base clip
# #                     base_clip = create_word_clip(word_data)
# #                     if base_clip is None:
# #                         continue
                        
# #                     word_width = base_clip.size[0]
                    
# #                     # Handle line breaks
# #                     if line_width + word_width > max_width:
# #                         x_pos = frame_size[0] * 0.1
# #                         y_pos += base_clip.size[1] + 10
# #                         line_width = 0
                    
# #                     # Position base clip
# #                     base_clip = base_clip.with_position((x_pos, y_pos))
# #                     current_line_clips.append(base_clip)
                    
# #                     # Create and position highlight clip
# #                     word_duration = word_data['end'] - word_data['start']
# #                     highlight_clip = create_word_clip(word_data, is_highlight=True)
# #                     if highlight_clip:
# #                         highlight_clip = (highlight_clip
# #                             .with_position((x_pos, y_pos))
# #                             .with_start(word_data['start'])
# #                             .with_duration(word_duration))
# #                         current_line_clips.append(highlight_clip)
                    
# #                     # Update positions
# #                     x_pos += word_width + 10
# #                     line_width += word_width + 10
                
# #                 # Only add clips if we have any for this line
# #                 if current_line_clips:
# #                     word_clips.extend(current_line_clips)

# #             print(f"Created {len(word_clips)} total clips")
# #             if not word_clips:
# #                 raise ValueError("No valid subtitle clips were created")
                
# #             return word_clips
            
# #         except Exception as e:
# #             logger.error(f"Error in create_word_level_subtitles: {str(e)}")
# #             print(f"Failed to create subtitle clips: {str(e)}")
# #             raise VideoProcessingError(f"Failed to create subtitle clips: {str(e)}")

# #     async def create_segment(self, segment: Dict, index: int, whisper_url: Optional[str] = None, 
# #                         session: Optional[aiohttp.ClientSession] = None) -> str:
# #         """Create a video segment with dynamically synchronized subtitles"""
# #         logger.info(f"Creating segment {index} with Whisper URL: {whisper_url}")
# #         print(f"\n=== Starting Segment {index} Creation ===")
# #         print(f"Whisper URL provided: {whisper_url}")
# #         final_clip = None
        
# #         if not whisper_url:
# #             logger.error("No Whisper URL provided")
# #             print("Error: Missing Whisper URL")
# #             raise VideoProcessingError("Whisper URL is required for subtitle generation")
        
# #         if not session:
# #             logger.error("No session provided")
# #             print("Error: Session is required")
# #             raise VideoProcessingError("Session is required for subtitle generation")
            
# #         try:
# #             print(f"Segment {index} data contains:")
# #             print(f"- Audio data length: {len(segment['audio_data']) if 'audio_data' in segment else 'Missing'}")
# #             print(f"- Image data length: {len(segment['image_data']) if 'image_data' in segment else 'Missing'}")
# #             print(f"- Story text length: {len(segment['story_text']) if 'story_text' in segment else 'Missing'}")
# #             print(f"Processing segment {index} audio and image...")
# #             logger.info("Processing audio and image files")
            
# #             audio_path = self._save_base64_audio(segment['audio_data'], index)
# #             print(f"Audio saved to: {audio_path}")
            
# #             image_array = self._decode_base64_image(segment['image_data'])
# #             print("Image decoded successfully")
            
# #             # Get synchronized subtitles if whisper_url is provided
# #             whisper_data = None
# #             if whisper_url and segment.get('audio_data'):
# #                 print(f"\nAttempting to get subtitles from Whisper API...")
# #                 logger.info(f"Whisper URL provided: {whisper_url}")
                
# #                 try:
# #                     whisper_data = await self.get_synchronized_subtitles(
# #                         segment['audio_data'],
# #                         whisper_url,
# #                         session
# #                     )
# #                     print("Successfully received whisper data")
# #                 except Exception as e:
# #                     logger.error(f"Subtitle generation failed: {str(e)}")
# #                     print(f"Failed to get subtitles: {str(e)}")
# #             else:
# #                 print("No Whisper URL provided or no audio data in segment")
# #                 logger.warning("Skipping subtitle generation - missing URL or audio data")
            
# #             # Create clips
# #             print("\nCreating video clips...")
# #             with AudioFileClip(audio_path) as audio_clip:
# #                 duration = audio_clip.duration
# #                 print(f"Audio duration: {duration} seconds")
                
# #                 video_clip = ImageClip(image_array).with_duration(duration)
# #                 video_with_audio = video_clip.with_audio(audio_clip)
                
# #                 if whisper_data and whisper_data.get('line_level'):
# #                     print("Creating subtitle clips...")
# #                     try:
# #                         # Create semi-transparent background
# #                         bg_clip = (ColorClip(size=video_clip.size, color=(64, 64, 64))
# #                             .with_opacity(0.6)
# #                             .with_duration(duration))
                        
# #                         # Create word-level subtitles with error checking
# #                         subtitle_clips = self.create_word_level_subtitles(
# #                             whisper_data,
# #                             video_clip.size,
# #                             duration
# #                         )
                        
# #                         if not subtitle_clips:
# #                             print("No subtitle clips were created, falling back to video without subtitles")
# #                             final_clip = video_with_audio
# #                         else:
# #                             print(f"Created {len(subtitle_clips)} subtitle clips")
# #                             # Combine everything
# #                             final_clip = CompositeVideoClip([
# #                                 video_with_audio,
# #                                 bg_clip.with_position(('center', 'bottom')),
# #                                 *subtitle_clips
# #                             ])
# #                             print("Composite video created with subtitles")
# #                     except Exception as e:
# #                         logger.error(f"Error creating subtitles: {str(e)}")
# #                         print(f"Falling back to video without subtitles due to error: {str(e)}")
# #                         final_clip = video_with_audio
# #                 else:
# #                     final_clip = video_with_audio
                
# #                 # Write segment
# #                 output_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
# #                 print(f"\nWriting video to: {output_path}")
                
# #                 final_clip.write_videofile(
# #                     output_path,
# #                     fps=24,
# #                     codec='libx264',
# #                     audio_codec='aac',
# #                     threads=4,
# #                     preset='medium',
# #                     remove_temp=True
# #                 )
                
# #                 self.segments.append(output_path)
# #                 print(f"Segment {index} completed successfully")
# #                 return output_path
                
# #         except Exception as e:
# #             logger.error(f"Segment {index} creation failed: {str(e)}")
# #             print(f"Error creating segment {index}: {str(e)}")
# #             raise VideoProcessingError(f"Failed to create segment {index}: {str(e)}")
# #         finally:
# #             print(f"=== Finishing Segment {index} Creation ===\n")
# #             if final_clip:
# #                 try:
# #                     final_clip.close()
# #                     print(f"Cleaned up resources for segment {index}")
# #                 except:
# #                     print(f"Warning: Could not clean up resources for segment {index}")
    
    
# #     def concatenate_segments(self) -> str:
# #         """Concatenate all segments into final video"""
# #         if not self.segments:
# #             raise VideoProcessingError("No segments to concatenate")

# #         try:
# #             clips = [VideoFileClip(path) for path in self.segments]
# #             final_video = concatenate_videoclips(clips)
# #             output_path = os.path.join(self.temp_dir, 'final_video.mp4')
            
# #             final_video.write_videofile(
# #                 output_path,
# #                 fps=30,
# #                 codec='libx264',
# #                 audio_codec='aac',
# #                 remove_temp=True
# #             )
            
# #             return output_path
            
# #         except Exception as e:
# #             raise VideoProcessingError(f"Failed to concatenate segments: {e}")
# #         finally:
# #             for clip in clips:
# #                 try:
# #                     clip.close()
# #                 except Exception:
# #                     pass

# #     def cleanup(self):
# #         """Clean up temporary files and directory"""
# #         try:
# #             # Remove segment files
# #             for segment in self.segments:
# #                 if os.path.exists(segment):
# #                     os.remove(segment)
            
# #             # Remove temp directory
# #             if os.path.exists(self.temp_dir):
# #                 import shutil
# #                 shutil.rmtree(self.temp_dir, ignore_errors=True)
                
# #         except Exception as e:
# #             logger.error(f"Cleanup error: {e}")

# #     def __enter__(self):
# #         return self

# #     def __exit__(self, exc_type, exc_val, exc_tb):
# #         self.cleanup()