import os
import base64
import tempfile
import logging
from typing import List, Optional, Dict
import numpy as np
from PIL import Image
import io
from datetime import timedelta
import json
import asyncio
import aiohttp
from moviepy import *
from moviepy.video.tools.subtitles import SubtitlesClip
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

class VideoManager:
    def __init__(self):
        """Initialize video manager with temporary directory and font settings"""
        self.temp_dir = tempfile.mkdtemp()
        self.segments: List[str] = []
        self.font_path = self._get_system_font()
        logger.info(f"VideoManager initialized with temp dir: {self.temp_dir}")

    def _get_system_font(self) -> str:
        """Get system font path based on OS"""
        font_paths = {
            'nt': [  # Windows
                r"C:\Windows\Fonts\Arial.ttf",
                r"C:\Windows\Fonts\Calibri.ttf",
                r"C:\Windows\Fonts\segoeui.ttf"
            ],
            'posix': [  # Linux/Unix
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/Arial.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ]
        }

        paths = font_paths.get(os.name, [])
        for path in paths:
            if os.path.exists(path):
                logger.info(f"Using system font: {path}")
                return path

        logger.warning("No system fonts found, using default")
        return ""

    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Convert base64 image to numpy array"""
        try:
            base64_str = base64_str.split('base64,')[-1]
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            return np.array(image.convert('RGB'))
        except Exception as e:
            raise VideoProcessingError(f"Failed to decode image: {e}")

    def _save_base64_audio(self, base64_str: str, index: int) -> str:
        """Save base64 audio to temporary WAV file"""
        try:
            base64_str = base64_str.split('base64,')[-1]
            audio_data = base64.b64decode(base64_str)
            audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            return audio_path
        except Exception as e:
            raise VideoProcessingError(f"Failed to save audio: {e}")

    async def get_synchronized_subtitles(self, audio_data: str, whisper_url: str, session: aiohttp.ClientSession) -> Dict:
        """Get synchronized subtitles for audio using Whisper API"""
        try:
            logger.info(f"Starting subtitle request to Whisper API at URL: {whisper_url}")
            print(f"Attempting to call Whisper API at: {whisper_url}")
            print(f"Audio data length before processing: {len(audio_data)}")
            if ',' in audio_data:
                audio_data = audio_data.split('base64,')[1]
                print(f"Audio data length after base64 split: {len(audio_data)}")
            logger.info("Preparing API request...")
            print("Preparing to send request to Whisper API...")
            try:
                async with session.post(
                    f"{whisper_url}/process_audio",
                    json={
                        "audio_data": audio_data,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=500)
                ) as response:
                    logger.info(f"Received response from Whisper API. Status: {response.status}")
                    print(f"Whisper API Response Status: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Whisper API error response: {error_text}")
                        print(f"Error from Whisper API: {error_text}")
                        raise VideoProcessingError(f"Whisper API error: {error_text}")
                    
                    logger.info("Successfully got response, parsing JSON...")
                    print("Parsing Whisper API response...")
                    
                    transcription_data = await response.json()
                    print("\n=== Whisper API Response Data ===")
                    print(json.dumps(transcription_data, indent=2))
                    print("===============================\n")
                    
                    if not transcription_data:
                        logger.error("Received empty transcription data")
                        print("Warning: Empty transcription data received")
                        raise VideoProcessingError("Empty transcription data received")
                    
                    if 'line_level' not in transcription_data:
                        logger.error(f"Missing line_level in response. Keys received: {transcription_data.keys()}")
                        print(f"Missing required data. Keys in response: {transcription_data.keys()}")
                        raise VideoProcessingError("Invalid transcription data: missing line_level")
                    
                    logger.info(f"Successfully processed transcription data with {len(transcription_data['line_level'])} lines")
                    print(f"Found {len(transcription_data['line_level'])} lines of transcription")
                    
                    return transcription_data
                        
            except aiohttp.ClientError as e:
                logger.error(f"Network error during API call: {str(e)}")
                print(f"Network error occurred: {str(e)}")
                raise VideoProcessingError(f"Network error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error in get_synchronized_subtitles: {str(e)}")
            print(f"Error getting subtitles: {str(e)}")
            raise VideoProcessingError(f"Failed to get synchronized subtitles: {str(e)}")

    def create_word_level_subtitles(self, whisper_data: Dict, frame_size: tuple, duration: float) -> List:
        """Creates word-level subtitle clips"""
        try:
            subtitle_clips = []
            words_data = whisper_data.get('word_level', [])
            
            for word_data in words_data:
                word = word_data['word'].strip()
                if not word:
                    continue
                word_clip = (TextClip(
                    text=word,
                    font=self.font_path,
                    font_size=int(frame_size[1] * 0.075), 
                    color='yellow',
                    stroke_color='black',
                    stroke_width=2
                )
                .with_position(('center', 'bottom'))
                .with_start(word_data['start'])
                .with_duration(word_data['end'] - word_data['start']))
                
                subtitle_clips.append(word_clip)
                
            return subtitle_clips
        except Exception as e:
            logger.error(f"Error creating word-level subtitles: {e}")
            return []

    async def create_segment(self, segment: Dict, index: int, whisper_url: Optional[str] = None, 
                        session: Optional[aiohttp.ClientSession] = None) -> str:
        """Create a video segment with dynamically synchronized subtitles"""
        logger.info(f"Creating segment {index} with Whisper URL: {whisper_url}")
        print(f"\n=== Starting Segment {index} Creation ===")
        print(f"Whisper URL provided: {whisper_url}")
        final_clip = None
        
        if not whisper_url:
            logger.error("No Whisper URL provided")
            print("Error: Missing Whisper URL")
            raise VideoProcessingError("Whisper URL is required for subtitle generation")
        
        if not session:
            logger.error("No session provided")
            print("Error: Session is required")
            raise VideoProcessingError("Session is required for subtitle generation")
            
        try:
            # Log input data
            print(f"Segment {index} data contains:")
            print(f"- Audio data length: {len(segment['audio_data']) if 'audio_data' in segment else 'Missing'}")
            print(f"- Image data length: {len(segment['image_data']) if 'image_data' in segment else 'Missing'}")
            print(f"- Story text length: {len(segment['story_text']) if 'story_text' in segment else 'Missing'}")
            
            # Create base video with audio
            print(f"Processing segment {index} audio and image...")
            logger.info("Processing audio and image files")
            
            audio_path = self._save_base64_audio(segment['audio_data'], index)
            print(f"Audio saved to: {audio_path}")
            
            image_array = self._decode_base64_image(segment['image_data'])
            print("Image decoded successfully")
            
            print("\nCreating video clips...")
            with AudioFileClip(audio_path) as audio_clip:
                duration = audio_clip.duration
                print(f"Audio duration: {duration} seconds")
                
                video_clip = ImageClip(image_array).with_duration(duration)
                video_with_audio = video_clip.with_audio(audio_clip)
                
                # Get and add subtitles if available
                try:
                    print(f"\nAttempting to get subtitles from Whisper API...")
                    logger.info(f"Whisper URL provided: {whisper_url}")
                    
                    whisper_data = await self.get_synchronized_subtitles(
                        segment['audio_data'],
                        whisper_url,
                        session
                    )
                    print("Successfully received whisper data")
                    
                    if whisper_data:
                        print("Creating subtitle clips...")
                        subtitle_clips = self.create_word_level_subtitles(
                            whisper_data,
                            video_clip.size,
                            duration
                        )
                        
                        if subtitle_clips:
                            print(f"Created {len(subtitle_clips)} subtitle clips")
                            final_clip = CompositeVideoClip([
                                video_with_audio,
                                *subtitle_clips
                            ])
                            print("Composite video created with subtitles")
                        else:
                            print("No subtitle clips were created, falling back to video without subtitles")
                            final_clip = video_with_audio
                    else:
                        print("No whisper data received, creating video without subtitles")
                        final_clip = video_with_audio
                        
                except Exception as e:
                    logger.error(f"Subtitle generation failed: {str(e)}")
                    print(f"Failed to create subtitles: {str(e)}")
                    final_clip = video_with_audio
                
                # Write output
                output_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
                print(f"\nWriting video to: {output_path}")
                
                final_clip.write_videofile(
                    output_path,
                    fps=24,
                    codec='libx264',
                    audio_codec='aac',
                    threads=4,
                    preset='medium',
                    remove_temp=True
                )
                
                self.segments.append(output_path)
                print(f"Segment {index} completed successfully")
                return output_path
                
        except Exception as e:
            logger.error(f"Segment {index} creation failed: {str(e)}")
            print(f"Error creating segment {index}: {str(e)}")
            raise VideoProcessingError(f"Failed to create segment {index}: {str(e)}")
        finally:
            print(f"=== Finishing Segment {index} Creation ===\n")
            if final_clip:
                try:
                    final_clip.close()
                    print(f"Cleaned up resources for segment {index}")
                except:
                    print(f"Warning: Could not clean up resources for segment {index}")
    
    
    def concatenate_segments(self) -> str:
        """Concatenate all segments into final video"""
        if not self.segments:
            raise VideoProcessingError("No segments to concatenate")

        try:
            clips = [VideoFileClip(path) for path in self.segments]
            final_video = concatenate_videoclips(clips)
            output_path = os.path.join(self.temp_dir, 'final_video.mp4')
            
            final_video.write_videofile(
                output_path,
                fps=30,
                codec='libx264',
                audio_codec='aac',
                remove_temp=True
            )
            
            return output_path
            
        except Exception as e:
            raise VideoProcessingError(f"Failed to concatenate segments: {e}")
        finally:
            for clip in clips:
                try:
                    clip.close()
                except Exception:
                    pass

    def cleanup(self):
        """Clean up temporary files and directory"""
        try:
            # Remove segment files
            for segment in self.segments:
                if os.path.exists(segment):
                    os.remove(segment)
            
            # Remove temp directory
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()








# def create_segment(self, segment: Dict, index: int) -> str:
    #     logger.info(f"Creating segment {index}")
        
    #     try:
    #         # Process audio and image
    #         audio_path = self._save_base64_audio(segment['audio_data'], index)
    #         image_array = self._decode_base64_image(segment['image_data'])
            
    #         with AudioFileClip(audio_path) as audio_clip:
    #             duration = audio_clip.duration
                
    #             # Create base video with proper size
    #             video_clip = ImageClip(image_array).with_duration(duration)
                
    #             # Resize if needed (assuming 1024x1024 input)
    #             if video_clip.size != (1024, 1024):
    #                 video_clip = video_clip.resize((1024, 1024))
                
    #             video_with_audio = video_clip.with_audio(audio_clip)
                
    #             if segment.get('story_text'):
    #                 # Create subtitle generator with improved positioning
    #                 def create_subtitle(txt):
    #                     return TextClip(
    #                         text=txt,
    #                         font=self.font_path,
    #                         font_size=50,
    #                         color='white',
    #                         stroke_color='black',
    #                         stroke_width=5,
    #                     ).with_position(('center', 0.65))  # Lower position (65% from top)
                    
    #                 # Improve subtitle timing
    #                 srt_content = self._create_srt_content(
    #                     segment['story_text'],
    #                     start=0,
    #                     duration=duration * 0.98  # Slight buffer at the end
    #                 )
                    
    #                 srt_path = os.path.join(self.temp_dir, f'sub_{index}.srt')
    #                 with open(srt_path, 'w', encoding='utf-8') as f:
    #                     f.write(srt_content)
                    
    #                 # Create subtitle clip with proper duration
    #                 subtitles = SubtitlesClip(
    #                     srt_path, 
    #                     make_textclip=create_subtitle
    #                 ).with_duration(duration)  # Match audio duration exactly

    #                 # Create final composite
    #                 final_clip = CompositeVideoClip(
    #                     [video_with_audio, subtitles],
    #                     size=video_with_audio.size
    #                 )
    #             else:
    #                 final_clip = video_with_audio

    #             # Write with higher quality settings
    #             output_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
    #             final_clip.write_videofile(
    #                 output_path,
    #                 fps=30,
    #                 codec='libx264',
    #                 audio_codec='aac',
    #                 bitrate='8000k',
    #                 preset='slower',  # Better quality
    #                 remove_temp=True
    #             )
                
    #             self.segments.append(output_path)
    #             return output_path

    #     except Exception as e:
    #         raise VideoProcessingError(f"Failed to create segment {index}: {e}")
    #     finally:
    #         if 'final_clip' in locals():
    #             final_clip.close()



# def _create_srt_content(self, text: str, start: float, duration: float) -> str:
    #     """Generate better formatted SRT content"""
    #     def format_time(seconds: float) -> str:
    #         td = timedelta(seconds=seconds)
    #         hours = td.seconds // 3600
    #         minutes = (td.seconds % 3600) // 60
    #         seconds = td.seconds % 60
    #         ms = round(td.microseconds / 1000)
    #         return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

    #     # Split long text into multiple subtitles if needed
    #     words = text.split()
    #     chunks = []
    #     current_chunk = []
        
    #     for word in words:
    #         current_chunk.append(word)
    #         if len(' '.join(current_chunk)) > 40:  # Max chars per line
    #             chunks.append(' '.join(current_chunk[:-1]))
    #             current_chunk = [word]
    #     if current_chunk:
    #         chunks.append(' '.join(current_chunk))

    #     # Create SRT entries
    #     srt_parts = []
    #     chunk_duration = duration / len(chunks)
        
    #     for i, chunk in enumerate(chunks, 1):
    #         chunk_start = start + (i-1) * chunk_duration
    #         chunk_end = chunk_start + chunk_duration
    #         srt_parts.append(
    #             f"{i}\n"
    #             f"{format_time(chunk_start)} --> {format_time(chunk_end)}\n"
    #             f"{chunk}\n"
    #         )

    #     return "\n".join(srt_parts)

    # @contextmanager
    # def _create_clips(self, image_array: np.ndarray, audio_path: str, duration: float):
    #     """Context manager for creating and cleaning up clips"""
    #     clips = []
    #     try:
    #         video_clip = ImageClip(image_array).with_duration(duration)
    #         audio_clip = AudioFileClip(audio_path)
    #         video_clip = video_clip.with_audio(audio_clip)
    #         clips.extend([video_clip, audio_clip])
    #         yield video_clip, audio_clip
    #     finally:
    #         for clip in clips:
    #             try:
    #                 clip.close()
    #             except Exception as e:
    #                 logger.error(f"Error closing clip: {e}")






# import os
# import base64
# import tempfile
# import logging
# from typing import List, Optional
# import numpy as np
# from PIL import Image
# import io
# from datetime import timedelta
# from moviepy import (
#     VideoFileClip, ImageClip, AudioFileClip, TextClip,
#     concatenate_videoclips, CompositeVideoClip, VideoClip
# )
# from moviepy.video.tools.subtitles import SubtitlesClip

# logger = logging.getLogger(__name__)

# class VideoProcessingError(Exception):
#     """Custom exception for video processing errors"""
#     pass

# class VideoManager:
#     def __init__(self):
#         """Initialize video manager with temporary directory"""
#         try:
#             self.temp_dir = tempfile.mkdtemp()
#             self.segments: List[str] = []
#             self.font_path = self._get_system_font()
#             logger.info(f"VideoManager initialized with temp directory: {self.temp_dir}")
#         except Exception as e:
#             raise VideoProcessingError(f"Failed to initialize VideoManager: {str(e)}")

#     def _get_system_font(self) -> str:
#         """Get a system font path based on OS"""
#         try:
#             if os.name == 'nt':  # Windows
#                 font_paths = [
#                     r"C:\Windows\Fonts\Arial.ttf",  # Note the capital A in Arial
#                     r"C:\Windows\Fonts\Calibri.ttf",
#                     r"C:\Windows\Fonts\segoeui.ttf"
#                 ]
#             else:  # Linux/Unix
#                 font_paths = [
#                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
#                     "/usr/share/fonts/TTF/Arial.ttf",
#                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
#                 ]

#             for path in font_paths:
#                 if os.path.exists(path):
#                     logger.info(f"Found and using system font: {path}")
#                     return path

#             # If no font is found, use default PIL font
#             logger.warning("No system fonts found, using default PIL font")
#             return ""  # Return an empty string to indicate no system font found
            
#         except Exception as e:
#             logger.error(f"Error finding system font: {str(e)}")
#             logger.warning("Using default PIL font")
#             return ""  # Let PIL use its default font

#     def _generate_srt_content(self, text: str, start_time: float, duration: float) -> str:
#         """Generate SRT format content for a single subtitle"""
#         try:
#             end_time = start_time + duration
            
#             def format_timedelta(seconds: float) -> str:
#                 """Convert seconds to SRT timestamp format"""
#                 td = timedelta(seconds=seconds)
#                 hours = td.seconds // 3600
#                 minutes = (td.seconds % 3600) // 60
#                 seconds = td.seconds % 60
#                 milliseconds = round(td.microseconds / 1000)
#                 return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

#             return f"1\n{format_timedelta(start_time)} --> {format_timedelta(end_time)}\n{text}\n\n"
            
#         except Exception as e:
#             logger.error(f"Error generating SRT content: {str(e)}")
#             raise VideoProcessingError(f"Failed to generate SRT content: {str(e)}")

#     def _save_srt_file(self, srt_content: str, index: int) -> str:
#         """Save SRT content to a temporary file"""
#         try:
#             srt_path = os.path.join(self.temp_dir, f'subtitle_{index}.srt')
#             with open(srt_path, 'w', encoding='utf-8') as f:
#                 f.write(srt_content)
#             return srt_path
#         except Exception as e:
#             logger.error(f"Error saving SRT file: {str(e)}")
#             raise VideoProcessingError(f"Failed to save SRT file: {str(e)}")

#     def _decode_base64_image(self, base64_str: str) -> np.ndarray:
#         """Convert base64 image to numpy array"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
            
#             image_data = base64.b64decode(base64_str)
#             image = Image.open(io.BytesIO(image_data))
            
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
                
#             logger.debug("Successfully decoded base64 image")
#             return np.array(image)
            
#         except Exception as e:
#             logger.error(f"Error decoding base64 image: {str(e)}")
#             raise VideoProcessingError(f"Failed to decode base64 image: {str(e)}")

#     def _save_base64_audio(self, base64_str: str, index: int) -> str:
#         """Save base64 audio to temporary WAV file"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
                
#             audio_data = base64.b64decode(base64_str)
#             audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
            
#             with open(audio_path, 'wb') as f:
#                 f.write(audio_data)
            
#             logger.debug(f"Successfully saved audio to {audio_path}")
#             return audio_path
            
#         except Exception as e:
#             logger.error(f"Error saving base64 audio: {str(e)}")
#             raise VideoProcessingError(f"Failed to save base64 audio: {str(e)}")

#     def create_segment(self, segment: dict, index: int) -> str:
#         """Create a video segment from image, audio and text"""
#         logger.info(f"Creating segment {index}")
        
#         audio_clip = None
#         video_clip = None
#         composite_clip = None
        
#         try:
#             # Step 1: Process Audio
#             logger.debug("Processing audio...")
#             audio_path = self._save_base64_audio(segment['audio_data'], index)
#             audio_clip = AudioFileClip(audio_path)
#             duration = audio_clip.duration
            
#             # Step 2: Process Image and create video
#             logger.debug("Processing image...")
#             image_array = self._decode_base64_image(segment['image_data'])
#             video_clip = ImageClip(image_array, duration=duration)
#             # video_clip = video_clip.set_fps(30)
#             # video_clip = video_clip.resize((1080, 1920))  # Set to vertical video dimensions
            
#             # Step 3: Create subtitles if story text is provided
#             if 'story_text' in segment and segment['story_text']:
#                 logger.debug("Creating subtitles...")
                
#                 # Generate and save SRT content
#                 srt_content = self._generate_srt_content(segment['story_text'], 0, duration)
#                 srt_path = self._save_srt_file(srt_content, index)
                
#                 # Create subtitle generator
#                 def make_textclip(txt):
#                     return TextClip(
#                         text=txt,
#                         font_size=40,
#                         color='white',
#                         stroke_color='black',
#                         stroke_width=5,
#                         font=self.font_path 
#                     )
                
#                 # Create subtitle clip without passing font parameter
#                 subtitles = SubtitlesClip(
#                     srt_path,
#                     make_textclip=make_textclip
#                 )
                
#                 # Combine video and subtitles
#                 composite_clip = CompositeVideoClip([
#                 video_clip,
#                 subtitles
#             ]).set_pos(('center', 'bottom'))
#                 composite_clip.duration = duration
#                 composite_clip.audio = audio_clip
                
#                 # Save segment with composite
#                 segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#                 logger.info(f"Writing segment to {segment_path}")
                
#                 composite_clip.write_videofile(
#                     segment_path,
#                     fps=30,
#                     codec='libx264',
#                     audio_codec='aac',
#                     remove_temp=True,
#                     logger=None
#                 )
#             else:
#                 # Save segment without subtitles
#                 segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#                 logger.info(f"Writing segment to {segment_path}")
                
#                 video_clip.set_audio(audio_clip).write_videofile(
#                     segment_path,
#                     fps=30,
#                     codec='libx264',
#                     audio_codec='aac',
#                     threads=4,
#                     remove_temp=True,
#                     logger=None
#                 )
            
#             self.segments.append(segment_path)
#             logger.info(f"Successfully created segment {index}")
#             return segment_path
            
#         except Exception as e:
#             logger.error(f"Error creating segment {index}: {str(e)}")
#             raise VideoProcessingError(f"Failed to create segment {index}: {str(e)}")
            
#         finally:
#             # Cleanup resources
#             for clip in [audio_clip, video_clip, composite_clip]:
#                 if clip is not None:
#                     try:
#                         clip.close()
#                     except Exception as e:
#                         logger.error(f"Error closing clip: {str(e)}")

#     def concatenate_segments(self) -> str:
#         """Concatenate segments"""
#         logger.info("Starting segment concatenation")
        
#         if not self.segments:
#             msg = "No segments to concatenate"
#             logger.error(msg)
#             raise VideoProcessingError(msg)
        
#         clips = []
#         try:
#             # Load all segment clips
#             logger.debug("Loading segments...")
#             clips = [VideoFileClip(path) for path in self.segments]
            
#             # Concatenate segments
#             logger.debug("Concatenating segments...")
#             final_video = concatenate_videoclips(clips, method="compose")
            
#             # Save final video
#             output_path = os.path.join(self.temp_dir, 'final_video.mp4')
#             logger.info(f"Writing final video to {output_path}")
            
#             final_video.write_videofile(
#                 output_path,
#                 fps=30,
#                 codec='libx264',
#                 audio_codec='aac',
#                 threads=4,
#                 remove_temp=True,
#                 logger=None
#             )
            
#             return output_path
            
#         except Exception as e:
#             logger.error(f"Error concatenating segments: {str(e)}")
#             raise VideoProcessingError(f"Failed to concatenate segments: {str(e)}")
            
#         finally:
#             # Cleanup clips
#             for clip in clips:
#                 try:
#                     clip.close()
#                 except Exception as e:
#                     logger.error(f"Error closing clip: {str(e)}")

#     def cleanup(self):
#         """Clean up temporary files and directory"""
#         logger.info("Starting cleanup")
        
#         try:
#             # Remove segment files
#             for segment in self.segments:
#                 try:
#                     if os.path.exists(segment):
#                         os.remove(segment)
#                         logger.debug(f"Removed segment: {segment}")
#                 except Exception as e:
#                     logger.error(f"Error removing segment {segment}: {str(e)}")
            
#             # Remove temp directory and its contents
#             if os.path.exists(self.temp_dir):
#                 try:
#                     import shutil
#                     shutil.rmtree(self.temp_dir, ignore_errors=True)
#                     logger.info(f"Removed temporary directory: {self.temp_dir}")
#                 except Exception as e:
#                     logger.error(f"Error removing temporary directory: {str(e)}")
                
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")














# import os
# import base64
# import tempfile
# import logging
# from typing import List, Optional
# import numpy as np
# from PIL import Image
# import io
# from moviepy import (
#     VideoFileClip, ImageClip, AudioFileClip, TextClip,
#     concatenate_videoclips, CompositeVideoClip, VideoClip, vfx
# )

# logger = logging.getLogger(__name__)

# class VideoProcessingError(Exception):
#     """Custom exception for video processing errors"""
#     pass

# class VideoManager:
#     def __init__(self):
#         """Initialize video manager with temporary directory"""
#         try:
#             self.temp_dir = tempfile.mkdtemp()
#             self.segments: List[str] = []
#             self.font_path = self._get_system_font()
#             logger.info(f"VideoManager initialized with temp directory: {self.temp_dir}")
#         except Exception as e:
#             raise VideoProcessingError(f"Failed to initialize VideoManager: {str(e)}")

#     def _get_system_font(self) -> str:
#         """Get a system font path based on OS"""
#         try:
#             if os.name == 'nt':  # Windows
#                 font_paths = [
#                     r"C:\Windows\Fonts\arial.ttf",
#                     r"C:\Windows\Fonts\calibri.ttf",
#                     r"C:\Windows\Fonts\segoeui.ttf"
#                 ]
#             else:  # Linux/Unix
#                 font_paths = [
#                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
#                     "/usr/share/fonts/TTF/arial.ttf",
#                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
#                 ]

#             for path in font_paths:
#                 if os.path.exists(path):
#                     logger.info(f"Using system font: {path}")
#                     return path

#             raise VideoProcessingError("No suitable system font found")
#         except Exception as e:
#             logger.error(f"Error finding system font: {str(e)}")
#             raise VideoProcessingError(f"Failed to find system font: {str(e)}")

#     def _decode_base64_image(self, base64_str: str) -> np.ndarray:
#         """Convert base64 image to numpy array"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
            
#             image_data = base64.b64decode(base64_str)
#             image = Image.open(io.BytesIO(image_data))
            
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
                
#             logger.debug("Successfully decoded base64 image")
#             return np.array(image)
            
#         except Exception as e:
#             logger.error(f"Error decoding base64 image: {str(e)}")
#             raise VideoProcessingError(f"Failed to decode base64 image: {str(e)}")

#     def _save_base64_audio(self, base64_str: str, index: int) -> str:
#         """Save base64 audio to temporary WAV file"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
                
#             audio_data = base64.b64decode(base64_str)
#             audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
            
#             with open(audio_path, 'wb') as f:
#                 f.write(audio_data)
            
#             logger.debug(f"Successfully saved audio to {audio_path}")
#             return audio_path
            
#         except Exception as e:
#             logger.error(f"Error saving base64 audio: {str(e)}")
#             raise VideoProcessingError(f"Failed to save base64 audio: {str(e)}")

#     def create_segment(self, segment: dict, index: int) -> str:
#         """Create a video segment from image, audio and text"""
#         logger.info(f"Creating segment {index}")
        
#         audio_clip = None
#         video_clip = None
#         text_clip = None
#         composite_clip = None
        
#         try:
#             # Step 1: Process Audio
#             logger.debug("Processing audio...")
#             audio_path = self._save_base64_audio(segment['audio_data'], index)
#             audio_clip = AudioFileClip(audio_path)
#             duration = audio_clip.duration
            
#             # Step 2: Process Image and create video
#             logger.debug("Processing image...")
#             image_array = self._decode_base64_image(segment['image_data'])
#             video_clip = VideoClip(lambda t: image_array)
#             video_clip.duration = duration
#             video_clip.audio = audio_clip
            
#             # Step 3: Create text overlay if story text is provided
#             if 'story_text' in segment and segment['story_text']:
#                 logger.debug("Creating text overlay...")
#                 text_clip = TextClip(
#                     text=segment['story_text'],
#                     font=self.font_path,
#                     font_size=40,
#                     size=(800, None),
#                     method='caption',
#                     color='white',
#                     bg_color=(0, 0, 0, 128),
#                     text_align='center',
#                     margin=(20, 20),
#                     transparent=True,
#                     duration=duration
#                 )
                
#                 # Combine video and text
#                 composite_clip = CompositeVideoClip([
#                     video_clip,
#                     # text_clip.set_position(('center', 'bottom'))
#                 ])
#                 composite_clip.duration = duration
#                 composite_clip.audio = audio_clip
                
#                 # Add CrossFadeIn effect if not the first clip
#                 if index > 0:
#                     composite_clip = composite_clip.with_effects([vfx.CrossFadeIn(1.0)])
                
#                 # Save segment with composite
#                 segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#                 logger.info(f"Writing segment to {segment_path}")
                
#                 composite_clip.write_videofile(
#                     segment_path,
#                     fps=24,
#                     codec='libx264',
#                     audio_codec='aac',
#                     remove_temp=True,
#                     logger=None
#                 )
#             else:
#                 # Add CrossFadeIn effect if not the first clip
#                 if index > 0:
#                     video_clip = video_clip.with_effects([vfx.CrossFadeIn(1.0)])
                
#                 # Save segment without text overlay
#                 segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#                 logger.info(f"Writing segment to {segment_path}")
                
#                 video_clip.write_videofile(
#                     segment_path,
#                     fps=24,
#                     codec='libx264',
#                     audio_codec='aac',
#                     remove_temp=True,
#                     logger=None
#                 )
            
#             self.segments.append(segment_path)
#             logger.info(f"Successfully created segment {index}")
#             return segment_path
            
#         except Exception as e:
#             logger.error(f"Error creating segment {index}: {str(e)}")
#             raise VideoProcessingError(f"Failed to create segment {index}: {str(e)}")
            
#         finally:
#             # Cleanup resources
#             for clip in [audio_clip, video_clip, text_clip, composite_clip]:
#                 if clip is not None:
#                     try:
#                         clip.close()
#                     except Exception as e:
#                         logger.error(f"Error closing clip: {str(e)}")

#     def concatenate_segments(self) -> str:
#         """Concatenate segments"""
#         logger.info("Starting segment concatenation")
        
#         if not self.segments:
#             msg = "No segments to concatenate"
#             logger.error(msg)
#             raise VideoProcessingError(msg)
        
#         clips = []
#         try:
#             # Load all segment clips
#             logger.debug("Loading segments...")
#             clips = [VideoFileClip(path) for path in self.segments]
            
#             # Concatenate segments
#             logger.debug("Concatenating segments...")
#             final_video = concatenate_videoclips(clips, method="compose")
            
#             # Save final video
#             output_path = os.path.join(self.temp_dir, 'final_video.mp4')
#             logger.info(f"Writing final video to {output_path}")
            
#             final_video.write_videofile(
#                 output_path,
#                 fps=24,
#                 codec='libx264',
#                 audio_codec='aac',
#                 remove_temp=True,
#                 logger=None
#             )
            
#             return output_path
            
#         except Exception as e:
#             logger.error(f"Error concatenating segments: {str(e)}")
#             raise VideoProcessingError(f"Failed to concatenate segments: {str(e)}")
            
#         finally:
#             # Cleanup clips
#             for clip in clips:
#                 try:
#                     clip.close()
#                 except Exception as e:
#                     logger.error(f"Error closing clip: {str(e)}")

#     def cleanup(self):
#         """Clean up temporary files and directory"""
#         logger.info("Starting cleanup")
        
#         try:
#             # Remove segment files
#             for segment in self.segments:
#                 try:
#                     if os.path.exists(segment):
#                         os.remove(segment)
#                         logger.debug(f"Removed segment: {segment}")
#                 except Exception as e:
#                     logger.error(f"Error removing segment {segment}: {str(e)}")
            
#             # Remove temp directory and its contents
#             if os.path.exists(self.temp_dir):
#                 try:
#                     import shutil
#                     shutil.rmtree(self.temp_dir, ignore_errors=True)
#                     logger.info(f"Removed temporary directory: {self.temp_dir}")
#                 except Exception as e:
#                     logger.error(f"Error removing temporary directory: {str(e)}")
                
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")


















# import os
# import base64
# import tempfile
# import logging
# from typing import List, Optional
# import numpy as np
# from PIL import Image, ImageFont
# import io
# from moviepy import (
#     VideoFileClip, ImageClip, AudioFileClip, TextClip,
#     concatenate_videoclips, CompositeVideoClip, VideoClip, vfx
# )

# logger = logging.getLogger(__name__)

# class VideoProcessingError(Exception):
#     """Custom exception for video processing errors"""
#     pass

# class VideoManager:
#     def __init__(self):
#         """Initialize video manager with temporary directory"""
#         try:
#             self.temp_dir = tempfile.mkdtemp()
#             self.segments: List[str] = []
#             # Find system font
#             self.font_path = self._get_system_font()
#             logger.info(f"VideoManager initialized with temp directory: {self.temp_dir}")
#         except Exception as e:
#             raise VideoProcessingError(f"Failed to initialize VideoManager: {str(e)}")

#     def _get_system_font(self) -> str:
#         """Get a system font path based on OS"""
#         try:
#             if os.name == 'nt':  # Windows
#                 font_paths = [
#                     r"C:\Windows\Fonts\arial.ttf",
#                     r"C:\Windows\Fonts\calibri.ttf",
#                     r"C:\Windows\Fonts\segoeui.ttf"
#                 ]
#             else:  # Linux/Unix
#                 font_paths = [
#                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
#                     "/usr/share/fonts/TTF/arial.ttf",
#                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
#                 ]

#             for path in font_paths:
#                 if os.path.exists(path):
#                     logger.info(f"Using system font: {path}")
#                     return path

#             raise VideoProcessingError("No suitable system font found")
#         except Exception as e:
#             logger.error(f"Error finding system font: {str(e)}")
#             raise VideoProcessingError(f"Failed to find system font: {str(e)}")

#     def _decode_base64_image(self, base64_str: str) -> np.ndarray:
#         """Convert base64 image to numpy array"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
            
#             image_data = base64.b64decode(base64_str)
#             image = Image.open(io.BytesIO(image_data))
            
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
                
#             logger.debug("Successfully decoded base64 image")
#             return np.array(image)
            
#         except Exception as e:
#             logger.error(f"Error decoding base64 image: {str(e)}")
#             raise VideoProcessingError(f"Failed to decode base64 image: {str(e)}")

#     def _save_base64_audio(self, base64_str: str, index: int) -> str:
#         """Save base64 audio to temporary WAV file"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
                
#             audio_data = base64.b64decode(base64_str)
#             audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
            
#             with open(audio_path, 'wb') as f:
#                 f.write(audio_data)
            
#             logger.debug(f"Successfully saved audio to {audio_path}")
#             return audio_path
            
#         except Exception as e:
#             logger.error(f"Error saving base64 audio: {str(e)}")
#             raise VideoProcessingError(f"Failed to save base64 audio: {str(e)}")

#     def create_segment(self, segment: dict, index: int) -> str:
#         """Create a video segment from image, audio and text"""
#         logger.info(f"Creating segment {index}")
        
#         audio_clip = None
#         video_clip = None
#         text_clip = None
#         composite_clip = None
        
#         try:
#             # Step 1: Process Audio
#             logger.debug("Processing audio...")
#             audio_path = self._save_base64_audio(segment['audio_data'], index)
#             audio_clip = AudioFileClip(audio_path)
#             duration = audio_clip.duration
            
#             # Step 2: Process Image and create video
#             logger.debug("Processing image...")
#             image_array = self._decode_base64_image(segment['image_data'])
#             video_clip = VideoClip(lambda t: image_array)
#             video_clip.duration = duration
#             video_clip.audio = audio_clip
            
#             # Step 3: Create text overlay if story text is provided
#             if 'story_text' in segment and segment['story_text']:
#                 logger.debug("Creating text overlay...")
#                 text_clip = TextClip(
#                     text=segment['story_text'],
#                     font=self.font_path,  # Use system font
#                     font_size=40,
#                     size=(800, None),
#                     method='caption',
#                     color='white',
#                     bg_color=(0, 0, 0, 128),
#                     text_align='center',
#                     margin=(20, 20),  # Add some margin around text
#                     transparent=True,
#                     duration=duration
#                 )
                
#                 # Combine video and text
#                 composite_clip = CompositeVideoClip([
#                     video_clip,
#                     # text_clip.set_position(('center', 'bottom'))
#                 ])
#                 composite_clip.duration = duration
#                 composite_clip.audio = audio_clip
                
#                 # Save segment with composite
#                 segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#                 logger.info(f"Writing segment to {segment_path}")
                
#                 composite_clip.write_videofile(
#                     segment_path,
#                     fps=24,
#                     codec='libx264',
#                     audio_codec='aac',
#                     remove_temp=True,
#                     logger=None
#                 )
#             else:
#                 # Save segment without text overlay
#                 segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#                 logger.info(f"Writing segment to {segment_path}")
                
#                 video_clip.write_videofile(
#                     segment_path,
#                     fps=24,
#                     codec='libx264',
#                     audio_codec='aac',
#                     remove_temp=True,
#                     logger=None
#                 )
            
#             self.segments.append(segment_path)
#             logger.info(f"Successfully created segment {index}")
#             return segment_path
            
#         except Exception as e:
#             logger.error(f"Error creating segment {index}: {str(e)}")
#             raise VideoProcessingError(f"Failed to create segment {index}: {str(e)}")
            
#         finally:
#             # Cleanup resources
#             for clip in [audio_clip, video_clip, text_clip, composite_clip]:
#                 if clip is not None:
#                     try:
#                         clip.close()
#                     except Exception as e:
#                         logger.error(f"Error closing clip: {str(e)}")

#     def concatenate_segments(self) -> str:
#         """Concatenate segments with crossfade transitions"""
#         logger.info("Starting segment concatenation")
        
#         if not self.segments:
#             msg = "No segments to concatenate"
#             logger.error(msg)
#             raise VideoProcessingError(msg)
        
#         clips = []
#         try:
#             # Load all segment clips
#             logger.debug("Loading segments...")
#             for i, path in enumerate(self.segments):
#                 clip = VideoFileClip(path)
#                 if i > 0:  # Add crossfade to all except first clip
#                     clip = clip.with_effects([vfx.crossfadein(1.0)])
#                 clips.append(clip)
            
#             # Concatenate with crossfading
#             logger.debug("Concatenating segments...")
#             final_video = concatenate_videoclips(
#                 clips,
#                 method="chain",
#                 padding=-1  # Negative padding for crossfade overlap
#             )
            
#             # Save final video
#             output_path = os.path.join(self.temp_dir, 'final_video.mp4')
#             logger.info(f"Writing final video to {output_path}")
            
#             final_video.write_videofile(
#                 output_path,
#                 fps=24,
#                 codec='libx264',
#                 audio_codec='aac',
#                 remove_temp=True,
#                 logger=None
#             )
            
#             return output_path
            
#         except Exception as e:
#             logger.error(f"Error concatenating segments: {str(e)}")
#             raise VideoProcessingError(f"Failed to concatenate segments: {str(e)}")
            
#         finally:
#             # Cleanup clips
#             for clip in clips:
#                 try:
#                     clip.close()
#                 except Exception as e:
#                     logger.error(f"Error closing clip: {str(e)}")

#     def cleanup(self):
#         """Clean up temporary files and directory"""
#         logger.info("Starting cleanup")
        
#         try:
#             # Remove segment files
#             for segment in self.segments:
#                 try:
#                     if os.path.exists(segment):
#                         os.remove(segment)
#                         logger.debug(f"Removed segment: {segment}")
#                 except Exception as e:
#                     logger.error(f"Error removing segment {segment}: {str(e)}")
            
#             # Remove temp directory and its contents
#             if os.path.exists(self.temp_dir):
#                 try:
#                     import shutil
#                     shutil.rmtree(self.temp_dir, ignore_errors=True)
#                     logger.info(f"Removed temporary directory: {self.temp_dir}")
#                 except Exception as e:
#                     logger.error(f"Error removing temporary directory: {str(e)}")
                
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")

















# import os
# import base64
# import tempfile
# import logging
# from typing import List, Optional
# import numpy as np
# from PIL import Image
# import io
# from moviepy import (
#     VideoFileClip, ImageClip, AudioFileClip, TextClip,
#     concatenate_videoclips, CompositeVideoClip, VideoClip, vfx
# )

# logger = logging.getLogger(__name__)

# class VideoProcessingError(Exception):
#     """Custom exception for video processing errors"""
#     pass

# class VideoManager:
#     def __init__(self):
#         """Initialize video manager with temporary directory"""
#         try:
#             self.temp_dir = tempfile.mkdtemp()
#             self.segments: List[str] = []
#             logger.info(f"VideoManager initialized with temp directory: {self.temp_dir}")
#         except Exception as e:
#             raise VideoProcessingError(f"Failed to initialize VideoManager: {str(e)}")

#     def _decode_base64_image(self, base64_str: str) -> np.ndarray:
#         """Convert base64 image to numpy array"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
            
#             image_data = base64.b64decode(base64_str)
#             image = Image.open(io.BytesIO(image_data))
            
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
                
#             logger.debug("Successfully decoded base64 image")
#             return np.array(image)
            
#         except Exception as e:
#             logger.error(f"Error decoding base64 image: {str(e)}")
#             raise VideoProcessingError(f"Failed to decode base64 image: {str(e)}")

#     def _save_base64_audio(self, base64_str: str, index: int) -> str:
#         """Save base64 audio to temporary WAV file"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
                
#             audio_data = base64.b64decode(base64_str)
#             audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
            
#             with open(audio_path, 'wb') as f:
#                 f.write(audio_data)
            
#             logger.debug(f"Successfully saved audio to {audio_path}")
#             return audio_path
            
#         except Exception as e:
#             logger.error(f"Error saving base64 audio: {str(e)}")
#             raise VideoProcessingError(f"Failed to save base64 audio: {str(e)}")

#     def create_segment(self, segment: dict, index: int) -> str:
#         """Create a video segment from image, audio and text"""
#         logger.info(f"Creating segment {index}")
        
#         audio_clip = None
#         video_clip = None
#         text_clip = None
#         composite_clip = None
        
#         try:
#             # Step 1: Process Audio
#             logger.debug("Processing audio...")
#             audio_path = self._save_base64_audio(segment['audio_data'], index)
#             audio_clip = AudioFileClip(audio_path)
#             duration = audio_clip.duration
            
#             # Step 2: Process Image and create video
#             logger.debug("Processing image...")
#             image_array = self._decode_base64_image(segment['image_data'])
#             video_clip = VideoClip(lambda t: image_array)
#             video_clip.duration = duration
#             video_clip.audio = audio_clip
            
#             # Step 3: Create text overlay if story text is provided
#             if 'story_text' in segment and segment['story_text']:
#                 logger.debug("Creating text overlay...")
#                 text_clip = TextClip(
#                     text=segment['story_text'],
#                     font=None,  # Uses Pillow default font
#                     font_size=40,
#                     size=(800, None),  # Fixed width, auto height
#                     method='caption',
#                     color='white',
#                     bg_color=(0, 0, 0, 128),  # Semi-transparent black
#                     text_align='center',
#                     horizontal_align='center',
#                     vertical_align='bottom',
#                     transparent=True,
#                     duration=duration
#                 )
                
#                 # Combine video and text
#                 composite_clip = CompositeVideoClip([
#                     video_clip,
#                     text_clip.set_position(('center', 'bottom'))
#                 ])
#                 composite_clip.duration = duration
#                 composite_clip.audio = audio_clip
                
#                 # Add crossfade if not first clip
#                 if index > 0:
#                     composite_clip = composite_clip.with_effects([vfx.crossfadein(1.0)])
                
#                 # Save segment with composite
#                 segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#                 logger.info(f"Writing segment to {segment_path}")
                
#                 composite_clip.write_videofile(
#                     segment_path,
#                     fps=24,
#                     codec='libx264',
#                     audio_codec='aac',
#                     remove_temp=True,
#                     logger=None
#                 )
#             else:
#                 # Save segment without text overlay
#                 segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#                 logger.info(f"Writing segment to {segment_path}")
                
#                 video_clip.write_videofile(
#                     segment_path,
#                     fps=24,
#                     codec='libx264',
#                     audio_codec='aac',
#                     remove_temp=True,
#                     logger=None
#                 )
            
#             self.segments.append(segment_path)
#             logger.info(f"Successfully created segment {index}")
#             return segment_path
            
#         except Exception as e:
#             logger.error(f"Error creating segment {index}: {str(e)}")
#             raise VideoProcessingError(f"Failed to create segment {index}: {str(e)}")
            
#         finally:
#             # Cleanup resources
#             for clip in [audio_clip, video_clip, text_clip, composite_clip]:
#                 if clip is not None:
#                     try:
#                         clip.close()
#                     except Exception as e:
#                         logger.error(f"Error closing clip: {str(e)}")

#     def concatenate_segments(self) -> str:
#         """Concatenate segments with crossfade transitions"""
#         logger.info("Starting segment concatenation")
        
#         if not self.segments:
#             msg = "No segments to concatenate"
#             logger.error(msg)
#             raise VideoProcessingError(msg)
        
#         clips = []
#         try:
#             # Load all segment clips
#             logger.debug("Loading segments...")
#             for i, path in enumerate(self.segments):
#                 clip = VideoFileClip(path)
#                 if i > 0:  # Add crossfade to all except first clip
#                     clip = clip.with_effects([vfx.crossfadein(1.0)])
#                 clips.append(clip)
            
#             # Concatenate with crossfading
#             logger.debug("Concatenating segments...")
#             final_video = concatenate_videoclips(
#                 clips,
#                 method="chain",
#                 padding=-1  # Negative padding for crossfade overlap
#             )
            
#             # Save final video
#             output_path = os.path.join(self.temp_dir, 'final_video.mp4')
#             logger.info(f"Writing final video to {output_path}")
            
#             final_video.write_videofile(
#                 output_path,
#                 fps=24,
#                 codec='libx264',
#                 audio_codec='aac',
#                 remove_temp=True,
#                 logger=None
#             )
            
#             return output_path
            
#         except Exception as e:
#             logger.error(f"Error concatenating segments: {str(e)}")
#             raise VideoProcessingError(f"Failed to concatenate segments: {str(e)}")
            
#         finally:
#             # Cleanup clips
#             for clip in clips:
#                 try:
#                     clip.close()
#                 except Exception as e:
#                     logger.error(f"Error closing clip: {str(e)}")

#     def cleanup(self):
#         """Clean up temporary files and directory"""
#         logger.info("Starting cleanup")
        
#         try:
#             # Remove segment files
#             for segment in self.segments:
#                 try:
#                     if os.path.exists(segment):
#                         os.remove(segment)
#                         logger.debug(f"Removed segment: {segment}")
#                 except Exception as e:
#                     logger.error(f"Error removing segment {segment}: {str(e)}")
            
#             # Remove temp directory and its contents
#             if os.path.exists(self.temp_dir):
#                 try:
#                     import shutil
#                     shutil.rmtree(self.temp_dir, ignore_errors=True)
#                     logger.info(f"Removed temporary directory: {self.temp_dir}")
#                 except Exception as e:
#                     logger.error(f"Error removing temporary directory: {str(e)}")
                
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")
#             # Don't raise here as this is cleanup code














# import os
# import base64
# import tempfile
# import logging
# from typing import List, Optional
# import numpy as np
# from PIL import Image
# import io
# from moviepy import VideoFileClip,ImageClip,AudioFileClip,concatenate_videoclips,CompositeVideoClip,VideoClip
# # from moviepy.video.VideoClip import ImageClip
# # from moviepy.audio.io.AudioFileClip import AudioFileClip
# # from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
# # from moviepy.video.compositing.concatenate import concatenate_videoclips

# logger = logging.getLogger(__name__)

# class VideoManager:
#     def __init__(self):
#         """Initialize video manager with temporary directory"""
#         self.temp_dir = tempfile.mkdtemp()
#         self.segments: List[str] = []  # Store paths to segment files
#         logger.info(f"VideoManager initialized with temp directory: {self.temp_dir}")

#     def _decode_base64_image(self, base64_str: str) -> np.ndarray:
#         """Convert base64 image to numpy array"""
#         try:
#             # Remove data URL prefix if present
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
            
#             # Decode base64 to image
#             image_data = base64.b64decode(base64_str)
#             image = Image.open(io.BytesIO(image_data))
            
#             # Convert to RGB if necessary
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
                
#             logger.debug("Successfully decoded base64 image")
#             return np.array(image)
            
#         except Exception as e:
#             logger.error(f"Error decoding base64 image: {str(e)}")
#             raise

#     def _save_base64_audio(self, base64_str: str, index: int) -> str:
#         """Save base64 audio to temporary WAV file"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
                
#             # Decode and save audio
#             audio_data = base64.b64decode(base64_str)
#             audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
            
#             with open(audio_path, 'wb') as f:
#                 f.write(audio_data)
            
#             logger.debug(f"Successfully saved audio to {audio_path}")
#             return audio_path
            
#         except Exception as e:
#             logger.error(f"Error saving base64 audio: {str(e)}")
#             raise

#     def create_segment(self, segment: dict, index: int) -> str:
#         """Create a video segment from image and audio"""
#         logger.info(f"Creating segment {index}")
        
#         audio_clip = None
#         image_clip = None
        
#         try:
#             # Step 1: Process Audio
#             logger.debug("Processing audio...")
#             audio_path = self._save_base64_audio(segment['audio_data'], index)
#             audio_clip = AudioFileClip(audio_path)
#             duration = audio_clip.duration
            
#             # Step 2: Process Image and create video
#             logger.debug("Processing image...")
#             image_array = self._decode_base64_image(segment['image_data'])
#             # Create ImageClip with duration
#             video_clip = VideoClip(lambda t: image_array)
#             video_clip.duration = duration
#             video_clip.audio = audio_clip
            
#             # Step 3: Save segment
#             segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#             logger.info(f"Writing segment to {segment_path}")
            
#             video_clip.write_videofile(
#                 segment_path,
#                 fps=24,
#                 codec='libx264',
#                 audio_codec='aac',
#                 remove_temp=True,
#                 logger=None
#             )
            
#             self.segments.append(segment_path)
#             logger.info(f"Successfully created segment {index}")
#             return segment_path
            
#         except Exception as e:
#             logger.error(f"Error creating segment {index}: {str(e)}")
#             raise
            
#         finally:
#             # Cleanup resources
#             if audio_clip:
#                 try:
#                     audio_clip.close()
#                 except Exception as e:
#                     logger.error(f"Error closing audio clip: {str(e)}")
#             if video_clip:
#                 try:
#                     video_clip.close()
#                 except Exception as e:
#                     logger.error(f"Error closing video clip: {str(e)}")

#     def concatenate_segments(self) -> str:
#         """Concatenate all segments into final video"""
#         logger.info("Starting segment concatenation")
        
#         if not self.segments:
#             msg = "No segments to concatenate"
#             logger.error(msg)
#             raise ValueError(msg)
        
#         clips = []
#         try:
#             # Load all segment clips
#             logger.debug("Loading segments...")
#             clips = [VideoFileClip(path) for path in self.segments]
            
#             # Concatenate
#             logger.debug("Concatenating segments...")
#             final_video = concatenate_videoclips(clips, method="chain")
            
#             # Save final video
#             output_path = os.path.join(self.temp_dir, 'final_video.mp4')
#             logger.info(f"Writing final video to {output_path}")
            
#             final_video.write_videofile(
#                 output_path,
#                 fps=24,
#                 codec='libx264',
#                 audio_codec='aac',
#                 remove_temp=True,
#                 logger=None
#             )
            
#             return output_path
            
#         except Exception as e:
#             logger.error(f"Error concatenating segments: {str(e)}")
#             raise
            
#         finally:
#             # Cleanup clips
#             for clip in clips:
#                 try:
#                     clip.close()
#                 except Exception as e:
#                     logger.error(f"Error closing clip: {str(e)}")

#     def cleanup(self):
#         """Clean up temporary files and directory"""
#         logger.info("Starting cleanup")
        
#         try:
#             # Remove segment files
#             for segment in self.segments:
#                 try:
#                     if os.path.exists(segment):
#                         os.remove(segment)
#                         logger.debug(f"Removed segment: {segment}")
#                 except Exception as e:
#                     logger.error(f"Error removing segment {segment}: {str(e)}")
            
#             # Remove temp directory
#             if os.path.exists(self.temp_dir):
#                 os.rmdir(self.temp_dir)
#                 logger.info(f"Removed temporary directory: {self.temp_dir}")
                
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")


# --------------------------------WORKS CODE-------------------------------------
# async def create_segment(self, segment: Dict, index: int, whisper_url: Optional[str] = None, session: Optional[aiohttp.ClientSession] = None) -> str:
    #     """Create a video segment with dynamically synchronized subtitles"""
    #     logger.info(f"Creating segment {index}")
    #     final_clip = None
        
    #     if not session:
    #         raise VideoProcessingError("Session is required for subtitle generation")
            
    #     try:
    #         # Process audio and image
    #         audio_path = self._save_base64_audio(segment['audio_data'], index)
    #         image_array = self._decode_base64_image(segment['image_data'])
            
    #         # Get synchronized subtitles if whisper_url is provided
    #         subtitle_data = None
    #         if whisper_url and segment.get('audio_data'):
    #             try:
    #                 subtitle_data = await self.get_synchronized_subtitles(
    #                     segment['audio_data'],
    #                     whisper_url,
    #                     session
    #                 )
    #             except Exception as e:
    #                 logger.error(f"Failed to get subtitles, continuing without them: {str(e)}")
            
    #         # Create clips
    #         audio_clip = AudioFileClip(audio_path)
    #         duration = audio_clip.duration
            
    #         # Create base video
    #         video_clip = ImageClip(image_array).with_duration(duration)
    #         video_with_audio = video_clip.with_audio(audio_clip)
            
    #         # Add subtitles if present
    #         if subtitle_data and subtitle_data.get('line_level'):
    #             def create_subtitle(txt):
    #                 return TextClip(
    #                     txt,
    #                     font=self.font_path or 'Arial',
    #                     fontsize=40,
    #                     color='white',
    #                     stroke_color='black',
    #                     stroke_width=2,
    #                     method='caption',
    #                     size=video_clip.size
    #                 ).with_position(('center', 'bottom'))
                
    #             # Create SRT content
    #             srt_path = os.path.join(self.temp_dir, f'sub_{index}.srt')
    #             with open(srt_path, 'w', encoding='utf-8') as f:
    #                 for i, line in enumerate(subtitle_data['line_level'], 1):
    #                     start_time = self._format_time(line['start'])
    #                     end_time = self._format_time(line['end'])
    #                     f.write(f"{i}\n{start_time} --> {end_time}\n{line['text']}\n\n")
                
    #             subtitles = SubtitlesClip(
    #                 srt_path,
    #                 make_textclip=create_subtitle
    #             )
                
    #             final_clip = CompositeVideoClip(
    #                 [video_with_audio, subtitles.with_duration(duration)],
    #                 size=video_with_audio.size
    #             )
    #         else:
    #             final_clip = video_with_audio
            
    #         # Write segment
    #         output_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
    #         final_clip.write_videofile(
    #             output_path,
    #             fps=24,
    #             codec='libx264',
    #             audio_codec='aac',
    #             threads=4,
    #             preset='medium',
    #             remove_temp=True
    #         )
            
    #         self.segments.append(output_path)
    #         return output_path
            
    #     except Exception as e:
    #         logger.error(f"Failed to create segment {index}: {str(e)}")
    #         raise VideoProcessingError(f"Failed to create segment {index}: {str(e)}")
    #     finally:
    #         # Clean up resources
    #         if final_clip:
    #             try:
    #                 final_clip.close()
    #             except:
    #                 pass
    
    
    
    
    
    
    
    
    
    
    # ------------------------------------------------
    
    # async def create_segment(self, segment: Dict, index: int, whisper_url: Optional[str] = None) -> str:
    #     """Create a video segment with dynamically synchronized subtitles"""
    #     logger.info(f"Creating segment {index}")
        
    #     try:
    #         # Process audio and image
    #         audio_path = self._save_base64_audio(segment['audio_data'], index)
    #         image_array = self._decode_base64_image(segment['image_data'])
            
    #         # Get synchronized subtitles if whisper_url is provided
    #         subtitle_data = None
    #         if whisper_url and segment.get('audio_data'):
    #             subtitle_data = await self.get_synchronized_subtitles(
    #                 segment['audio_data'], 
    #                 whisper_url
    #             )
            
    #         with AudioFileClip(audio_path) as audio_clip:
    #             duration = audio_clip.duration
                
    #             # Create base video
    #             video_clip = ImageClip(image_array).with_duration(duration)
    #             video_with_audio = video_clip.with_audio(audio_clip)
                
    #             # Add subtitles if present
    #             if subtitle_data and subtitle_data.get('line_level'):
    #                 def create_subtitle(txt):
    #                     return TextClip(
    #                         text=txt,
    #                         font=self.font_path,
    #                         font_size=50, 
    #                         color='white',
    #                         stroke_color='black',
    #                         stroke_width=5,
    #                     ).with_position(('center', 0.8))
                    
    #                 # Create SRT content from Whisper's line-level data
    #                 srt_content = ""
    #                 for i, line in enumerate(subtitle_data['line_level'], 1):
    #                     start_time = self._format_time(line['start'])
    #                     end_time = self._format_time(line['end'])
    #                     srt_content += f"{i}\n{start_time} --> {end_time}\n{line['text']}\n\n"
                    
    #                 srt_path = os.path.join(self.temp_dir, f'sub_{index}.srt')
    #                 with open(srt_path, 'w', encoding='utf-8') as f:
    #                     f.write(srt_content)
                    
    #                 # Create subtitle clip with precise timing
    #                 subtitles = SubtitlesClip(
    #                     srt_path, 
    #                     make_textclip=create_subtitle
    #                 ).with_duration(duration)
                    
    #                 # Composite with proper layering
    #                 final_clip = CompositeVideoClip(
    #                     [video_with_audio, subtitles],
    #                     size=video_with_audio.size
    #                 )
    #             else:
    #                 final_clip = video_with_audio
                
    #             # Write segment
    #             output_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
    #             final_clip.write_videofile(
    #                 output_path,
    #                 fps=30,
    #                 codec='libx264',
    #                 audio_codec='aac',
    #                 bitrate='8000k',
    #                 remove_temp=True
    #             )
                
    #             self.segments.append(output_path)
    #             return output_path
                
    #     except Exception as e:
    #         raise VideoProcessingError(f"Failed to create segment {index}: {e}")
    #     finally:
    #         if 'final_clip' in locals():
    #             final_clip.close()
    
    
    # def create_segment(self, segment: Dict, index: int) -> str:
    #     """Create a video segment with dynamic subtitles"""
    #     logger.info(f"Creating segment {index}")
        
    #     try:
    #         # Process audio and image
    #         audio_path = self._save_base64_audio(segment['audio_data'], index)
    #         image_array = self._decode_base64_image(segment['image_data'])
            
    #         with AudioFileClip(audio_path) as audio_clip:
    #             duration = audio_clip.duration
                
    #             # Create base video
    #             video_clip = ImageClip(image_array).with_duration(duration)
    #             video_with_audio = video_clip.with_audio(audio_clip)
                
    #             # Add subtitles if present
    #             if segment.get('story_text'):
    #                 # Create subtitle generator - key change from old code
    #                 def create_subtitle(txt):
    #                     return TextClip(
    #                         text=txt,
    #                         font=self.font_path,
    #                         font_size=50, 
    #                         color='white',
    #                         stroke_color='black',
    #                         stroke_width=5,
    #                     ).with_position(('center', 0.8))  # Position relative to frame height
                        
    #                 # Create subtitle file
    #                 srt_content = self._create_srt_content(
    #                     segment['story_text'], 0, duration)
    #                 srt_path = os.path.join(self.temp_dir, f'sub_{index}.srt')
    #                 with open(srt_path, 'w', encoding='utf-8') as f:
    #                     f.write(srt_content)
                    
    #                 # Create subtitle clip with generator
    #                 subtitles = SubtitlesClip(
    #                     srt_path, 
    #                     make_textclip=create_subtitle
    #                 ).with_duration(duration)

    #                 # Composite with proper layering
    #                 final_clip = CompositeVideoClip(
    #                     [video_with_audio, subtitles],
    #                     size=video_with_audio.size
    #                 )
    #             else:
    #                 final_clip = video_with_audio

    #             # Write segment with improved settings
    #             output_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
    #             final_clip.write_videofile(
    #                 output_path,
    #                 fps=30,
    #                 codec='libx264',
    #                 audio_codec='aac',
    #                 bitrate='8000k',  # Higher bitrate for quality
    #                 remove_temp=True
    #             )
                
    #             self.segments.append(output_path)
    #             return output_path

    #     except Exception as e:
    #         raise VideoProcessingError(f"Failed to create segment {index}: {e}")
    #     finally:
    #         if 'final_clip' in locals():
    #             final_clip.close()
    
    
    
    
    
    
    
    
    # async def get_synchronized_subtitles(self, audio_data: str, whisper_url: str) -> Dict:
    #     """Get synchronized subtitles for audio using Whisper API"""
    #     try:
    #         logger.info("Getting synchronized subtitles from Whisper API")
            
    #         async with aiohttp.ClientSession() as session:
    #             async with session.post(
    #                 f"{whisper_url}/audio-process",
    #                 json={"audio_data": audio_data,
    #                       "type": "transcribe"},
    #                 headers={"Content-Type": "application/json"}
    #             ) as response:
    #                 if response.status != 200:
    #                     raise VideoProcessingError(f"Whisper API error: {await response.text()}")
                    
    #                 transcription_data = await response.json()
    #                 logger.info("Received transcription data from Whisper API")
    #                 return transcription_data
                    
    #     except Exception as e:
    #         raise VideoProcessingError(f"Failed to get synchronized subtitles: {e}")
    
    
    
    
    
    
    
    
    
    # async def create_segment(self, segment: Dict, index: int, whisper_url: Optional[str] = None, 
    #                         session: Optional[aiohttp.ClientSession] = None) -> str:
    #     """Create a video segment with dynamically synchronized subtitles"""
    #     logger.info(f"Creating segment {index}")
    #     final_clip = None
        
    #     if not session:
    #         raise VideoProcessingError("Session is required for subtitle generation")
            
    #     try:
    #         # Process audio and image
    #         audio_path = self._save_base64_audio(segment['audio_data'], index)
    #         image_array = self._decode_base64_image(segment['image_data'])
            
    #         # Get synchronized subtitles if whisper_url is provided
    #         whisper_data = None
    #         if whisper_url and segment.get('audio_data'):
    #             try:
    #                 whisper_data = await self.get_synchronized_subtitles(
    #                     segment['audio_data'],
    #                     whisper_url,
    #                     session
    #                 )
    #             except Exception as e:
    #                 logger.error(f"Failed to get subtitles, continuing without them: {str(e)}")
            
    #         # Create clips
    #         with AudioFileClip(audio_path) as audio_clip:
    #             duration = audio_clip.duration
    #             video_clip = ImageClip(image_array).with_duration(duration)
    #             video_with_audio = video_clip.with_audio(audio_clip)
                
    #             if whisper_data and whisper_data.get('line_level'):
    #                 # Create semi-transparent background for subtitles
    #                 bg_clip = (ColorClip(size=video_clip.size, color=(64, 64, 64))
    #                     .with_opacity(0.6)
    #                     .with_duration(duration))
                    
    #                 # Create word-level subtitles
    #                 subtitle_clips = self.create_word_level_subtitles(
    #                     whisper_data,
    #                     video_clip.size,
    #                     duration
    #                 )
                    
    #                 # Combine everything
    #                 final_clip = CompositeVideoClip([
    #                     video_with_audio,
    #                     bg_clip.with_position(('center', 'bottom')),
    #                     *subtitle_clips
    #                 ])
    #             else:
    #                 final_clip = video_with_audio
                
    #             # Write segment
    #             output_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
    #             final_clip.write_videofile(
    #                 output_path,
    #                 fps=24,
    #                 codec='libx264',
    #                 audio_codec='aac',
    #                 threads=4,
    #                 preset='medium',
    #                 remove_temp=True
    #             )
                
    #             self.segments.append(output_path)
    #             return output_path
                
    #     except Exception as e:
    #         logger.error(f"Failed to create segment {index}: {str(e)}")
    #         raise VideoProcessingError(f"Failed to create segment {index}: {str(e)}")
    #     finally:
    #         # Clean up resources
    #         if final_clip:
    #             try:
    #                 final_clip.close()
    #             except:
    #                 pass