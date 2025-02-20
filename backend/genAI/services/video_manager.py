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
            
            # Method 1: Using relative positioning
            # Position at center horizontally, and 90% down the frame vertically
            position = ('center', 0.78)
            relative = True

            # Method 2: Using absolute positioning with calculation
            # bottom_offset = int(frame_size[1] * 0.1)  # 10% from bottom
            # position = ('center', frame_size[1] - bottom_offset)
            
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
                .with_position(position, relative=relative)
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
            print(f"Segment {index} data contains:")
            print(f"- Audio data length: {len(segment['audio_data']) if 'audio_data' in segment else 'Missing'}")
            print(f"- Image data length: {len(segment['image_data']) if 'image_data' in segment else 'Missing'}")
            print(f"- Story text length: {len(segment['story_text']) if 'story_text' in segment else 'Missing'}")
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