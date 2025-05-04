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
        self.transition_duration = 1.0
        logger.info(f"VideoManager initialized with temp dir: {self.temp_dir}")

    def _get_system_font(self) -> str:
        """Get system font path based on OS"""
        font_paths = {
            'nt': [  
                r"C:\Windows\Fonts\Arial.ttf",
                r"C:\Windows\Fonts\Calibri.ttf",
                r"C:\Windows\Fonts\segoeui.ttf"
            ],
            'posix': [  
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


    def create_word_level_subtitles(self, whisper_data: Dict, frame_size: tuple, duration: float, subtitle_color: str = "#ff00ff") -> List:
        """Creates word-level subtitle clips with customizable color"""
        try:
            subtitle_clips = []
            words_data = whisper_data.get('word_level', [])
            position = ('center', 0.78)
            relative = True
            
            # Use the provided subtitle color
            color = subtitle_color
            logger.info(f"Using subtitle color: {color}")
            
            for word_data in words_data:
                word = word_data['word'].strip()
                if not word:
                    continue
                    
                word_clip = (TextClip(
                    text=word,
                    font=self.font_path,
                    font_size=int(frame_size[1] * 0.075), 
                    color=color,  # Dynamic color from parameter
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
            print(f"- Subtitle color: {segment.get('subtitle_color', '#ff00ff')}")
            
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
                        # Get subtitle color from segment, default to pink if not specified
                        subtitle_color = segment.get('subtitle_color', '#ff00ff')
                        
                        subtitle_clips = self.create_word_level_subtitles(
                            whisper_data,
                            video_clip.size,
                            duration,
                            subtitle_color=subtitle_color  # Pass subtitle color to method
                        )
                        
                        if subtitle_clips:
                            print(f"Created {len(subtitle_clips)} subtitle clips with color {subtitle_color}")
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
    
    
    def concatenate_segments(self, background_audio_path, split_video_path) -> str:
        """Concatenate all segments into final video with optional background video and audio"""
        if not self.segments:
            raise VideoProcessingError("No segments to concatenate")

        clips = []
        try:
            # Process segments with transitions
            for i, path in enumerate(self.segments):
                clip = VideoFileClip(path)
                
                if i == 0:
                    clip = clip.with_effects([
                        vfx.FadeOut(self.transition_duration)
                    ])
                elif i == len(self.segments) - 1:
                    clip = clip.with_effects([
                        vfx.FadeIn(self.transition_duration)
                    ])
                else:
                    clip = clip.with_effects([
                        vfx.FadeIn(self.transition_duration),
                        vfx.FadeOut(self.transition_duration)
                    ])
                clips.append(clip)

            # Create initial video by concatenating the clips
            generated_video = concatenate_videoclips(
                clips,
                method="compose"  
            )
            
            # Check if we should include background video (split screen)
            use_background_video = True
            if split_video_path == "NONE":
                logger.info("No background video requested, using only generated content")
                use_background_video = False
                final_video = generated_video
            elif not os.path.exists(split_video_path):
                logger.warning(f"Split video file not found at {split_video_path}, using only generated video")
                use_background_video = False
                final_video = generated_video
            else:
                # Include background video (split screen)
                try:
                    logger.info(f"Loading split video from: {split_video_path}")
                    split_video = VideoFileClip(split_video_path)
                    logger.info(f"Successfully loaded split video with duration: {split_video.duration}s")
                    
                    split_video = split_video.subclipped(0, generated_video.duration)
                    logger.info(f"Trimmed split video to match generated video duration: {generated_video.duration}s")
                    
                    target_width = min(generated_video.size[0], split_video.size[0])
                    generated_video_resized = generated_video.resized(width=target_width)
                    split_video_resized = split_video.resized(width=target_width)
                    
                    logger.info(f"Resized videos to WIDTH: {target_width}px")
                    
                    logger.info("Creating split screen video...")
                    final_video = clips_array([
                        [generated_video_resized],
                        [split_video_resized]
                    ])
                    logger.info(f"Successfully created split screen video with size: {final_video.size}")
                except Exception as e:
                    logger.error(f"Error creating split screen: {str(e)}")
                    logger.exception("Split screen creation failed, using only generated video")
                    final_video = generated_video
            
            # Add background audio if requested and available
            if background_audio_path == "NONE":
                logger.info("No background audio requested, using only original audio")
            else:
                background_audio = None
                if os.path.exists(background_audio_path):
                    try:
                        logger.info(f"Loading background audio from: {background_audio_path}")
                        background_audio = AudioFileClip(background_audio_path)
                        logger.info(f"Successfully loaded background audio with duration: {background_audio.duration}s")
                        
                        if background_audio.duration > final_video.duration:
                            logger.info(f"Trimming background audio to match video duration ({final_video.duration}s)")
                            background_audio = background_audio.subclipped(0, final_video.duration)
                            logger.info(f"New background audio duration: {background_audio.duration}s")
                        
                        background_audio_adjusted = background_audio.with_effects([afx.MultiplyVolume(0.5)])
                        
                        if final_video.audio is not None:
                            original_audio = final_video.audio
                            mixed_audio = CompositeAudioClip([
                                original_audio, 
                                background_audio_adjusted 
                            ])
                        else:
                            mixed_audio = background_audio_adjusted
                        
                        final_video = final_video.with_audio(mixed_audio)
                        logger.info("Background music added to final video")
                    except Exception as e:
                        logger.error(f"Error applying background audio: {str(e)}")
                        logger.exception("Detailed stacktrace:")
                else:
                    logger.warning(f"Background audio file not found at {background_audio_path}")
            
            # Write final video file
            output_path = os.path.join(self.temp_dir, 'final_video.mp4')
            
            final_video.write_videofile(
                output_path,
                fps=30,
                codec='libx264',
                audio_codec='aac',
                remove_temp=True,
                threads=4,
                preset='medium'
            )
            
            return output_path

        except Exception as e:
            logger.error(f"Failed to concatenate segments: {str(e)}")
            logger.exception("Detailed stacktrace:")
            raise VideoProcessingError(f"Failed to concatenate segments: {e}")
        finally:
            # Clean up resources
            for clip in clips:
                try:
                    clip.close()
                except Exception:
                    logger.warning(f"Failed to close clip")
            
            if 'generated_video' in locals():
                try:
                    generated_video.close()
                except Exception:
                    logger.warning("Failed to close generated video")
                    
            if 'split_video' in locals() and 'split_video' in vars() and split_video is not None:
                try:
                    split_video.close()
                except Exception:
                    logger.warning("Failed to close split video")
                    
            if 'final_video' in locals():
                try:
                    final_video.close()
                except Exception:
                    logger.warning("Failed to close final video")
                    
            if 'background_audio' in locals() and 'background_audio' in vars() and background_audio is not None:
                try:
                    background_audio.close()
                except Exception:
                    logger.warning("Failed to close background audio")
                    
    async def create_video_segment(self, segment: Dict, index: int, whisper_url: Optional[str] = None,
                                   session: Optional[aiohttp.ClientSession] = None) -> str:
        """Create a video segment with dynamically synchronized subtitles from a WAN-generated video"""
        logger.info(f"Creating video segment {index} with Whisper URL: {whisper_url}")
        print(f"\n=== Starting Video Segment {index} Creation ===")
        print(f"Whisper URL provided: {whisper_url}")

        # --- Resource Management Variables ---
        video_clip = None  # Will hold the VideoFileClip object
        final_clip = None  # Will hold the final CompositeVideoClip or equivalent
        # audio_clip is handled by 'with' statement below
        # ---

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
            print(f"- Video path: {segment.get('video_path', 'Missing')}")
            print(f"- Story text length: {len(segment['story_text']) if 'story_text' in segment else 'Missing'}")
            print(f"- Subtitle color: {segment.get('subtitle_color', '#ff00ff')}")

            audio_path = self._save_base64_audio(segment['audio_data'], index)
            print(f"Audio saved to: {audio_path}")

            # Load the video file
            video_path = segment.get('video_path') # Use .get for safety
            if not video_path or not os.path.exists(video_path):
                 raise VideoProcessingError(f"Video file not found or not specified: {video_path}")

            print(f"Loading video from: {video_path}")
            # Load the video clip *before* the audio 'with' block if possible,
            # but keep track of it for closing.
            video_clip = VideoFileClip(video_path)
            print(f"Original video duration: {video_clip.duration}s, FPS: {video_clip.fps}")


            print("\nProcessing clips...")
            with AudioFileClip(audio_path) as audio_clip:
                duration = audio_clip.duration
                print(f"Audio duration: {duration} seconds")

                # Keep a reference to the potentially modified clip
                processed_video_clip = video_clip

                # Adjust video duration to match audio if needed
                if video_clip.duration < duration:
                    print(f"Video duration ({video_clip.duration}s) is shorter than audio, looping video")
                    # 1. Create an instance of the Loop effect class
                    loop_effect_instance = vfx.Loop(duration=duration)

                    # 2. Apply the effect instance using with_effects (pass a LIST of effects)
                    processed_video_clip = video_clip.with_effects([loop_effect_instance])
                    # --- End MoviePy v2.0 Syntax ---

                elif video_clip.duration > duration:
                    print(f"Video duration ({video_clip.duration}s) is longer than audio, trimming video")
                    processed_video_clip = video_clip.subclipped(0, duration)
                else:
                     print("Video and audio durations match.")
                     # No change needed, processed_video_clip remains the same as video_clip

                print(f"Adjusted video duration: {processed_video_clip.duration}s")

                # Replace original audio with narration using the processed clip
                video_with_audio = processed_video_clip.with_audio(audio_clip)

                try:
                    print(f"\nAttempting to get subtitles from Whisper API...")
                    logger.info(f"Whisper URL provided: {whisper_url}")

                    whisper_data = await self.get_synchronized_subtitles(
                        segment['audio_data'],
                        whisper_url,
                        session
                    )


                    if whisper_data:
                        print("Successfully received whisper data")
                        print("Creating subtitle clips...")
                        subtitle_color = segment.get('subtitle_color', '#ff00ff')

                        subtitle_clips = self.create_word_level_subtitles(
                            whisper_data,
                            processed_video_clip.size, # Use size from the processed clip
                            duration,
                            subtitle_color=subtitle_color
                        )

                        if subtitle_clips:
                            print(f"Created {len(subtitle_clips)} subtitle clips with color {subtitle_color}")
                            # Assign to final_clip
                            final_clip = CompositeVideoClip([
                                video_with_audio,
                                *subtitle_clips
                            ])
                            print("Composite video created with subtitles")
                        else:
                            print("No subtitle clips were created, falling back to video without subtitles")
                            final_clip = video_with_audio # Assign fallback to final_clip
                    else:
                        print("No whisper data received, creating video without subtitles")
                        final_clip = video_with_audio # Assign fallback to final_clip

                except Exception as e:
                    logger.error(f"Subtitle generation failed: {str(e)}", exc_info=True)
                    print(f"Failed to create subtitles: {str(e)}")
                    final_clip = video_with_audio # Assign fallback to final_clip

                output_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
                print(f"\nWriting video to: {output_path}")

                # Ensure final_clip is assigned before writing
                if not final_clip:
                     logger.error("Final clip was not assigned before writing.")
                     # This case should ideally be covered by fallbacks above, but check for safety
                     raise VideoProcessingError("Failed to create final video clip.")

                # Use source FPS if available, otherwise default to 24
                output_fps = processed_video_clip.fps if processed_video_clip.fps else 24
                print(f"Using FPS for writing: {output_fps}")

                final_clip.write_videofile(
                    output_path,
                    fps=output_fps, # Use detected or default FPS
                    codec='libx264',
                    audio_codec='aac',
                    threads=4, # Adjust based on your system
                    preset='medium', # Faster presets: 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast'
                    remove_temp=True,
                    logger=None # Correct for newer MoviePy versions
                )

                self.segments.append(output_path)
                print(f"Segment {index} completed successfully: {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"Video segment {index} creation failed: {str(e)}", exc_info=True)
            print(f"Error creating video segment {index}: {str(e)}")
            if isinstance(e, VideoProcessingError):
                 raise
            else:
                 raise VideoProcessingError(f"Failed to create video segment {index}: {str(e)}") from e                
    
    
    def concatenate_wan_videos(self, video_paths: List[str]) -> str:
        """
        Concatenate multiple videos into a single video file without any 
        transitions or effects - just simple concatenation.
        
        Args:
            video_paths: List of paths to video files
            
        Returns:
            Path to the concatenated video file
        """
        try:
            if not video_paths:
                raise ValueError("No video paths provided")
                
            logger.info(f"Concatenating {len(video_paths)} videos")
                
            # Load video clips
            clips = []
            for i, path in enumerate(video_paths):
                logger.info(f"Loading video clip {i+1}/{len(video_paths)}: {path}")
                clip = VideoFileClip(path)
                clips.append(clip)
                
            # Concatenate clips (simple concatenation, no transitions)
            logger.info("Performing basic concatenation without transitions")
            final_clip = concatenate_videoclips(clips, method="chain")
            
            # Save to temporary file
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            logger.info(f"Writing concatenated video to: {output_path}")
            
            # Remove verbose parameter and use logger=None instead
            final_clip.write_videofile(output_path, codec="libx264", logger=None)
            
            # Close clips to free resources
            for clip in clips:
                clip.close()
            final_clip.close()
            
            logger.info("Video concatenation complete")
            return output_path
            
        except Exception as e:
            logger.error(f"Error concatenating videos: {str(e)}")
            raise
        
        
    
    
    # def concatenate_wan_videos(self, video_paths: List[str]) -> str:
    #     """
    #     Concatenate multiple videos into a single video file without any 
    #     transitions or effects - just simple concatenation.
        
    #     Args:
    #         video_paths: List of paths to video files
            
    #     Returns:
    #         Path to the concatenated video file
    #     """
    #     try:
    #         if not video_paths:
    #             raise ValueError("No video paths provided")
                
    #         logger.info(f"Concatenating {len(video_paths)} videos")
                
    #         # Load video clips
    #         clips = []
    #         for i, path in enumerate(video_paths):
    #             logger.info(f"Loading video clip {i+1}/{len(video_paths)}: {path}")
    #             clip = VideoFileClip(path)
    #             clips.append(clip)
                
    #         # Concatenate clips (simple concatenation, no transitions)
    #         logger.info("Performing basic concatenation without transitions")
    #         final_clip = concatenate_videoclips(clips, method="chain")
            
    #         # Save to temporary file
    #         output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    #         logger.info(f"Writing concatenated video to: {output_path}")
    #         final_clip.write_videofile(output_path, codec="libx264", verbose=False, logger=None)
            
    #         # Close clips to free resources
    #         for clip in clips:
    #             clip.close()
    #         final_clip.close()
            
    #         logger.info("Video concatenation complete")
    #         return output_path
            
    #     except Exception as e:
    #         logger.error(f"Error concatenating videos: {str(e)}")
    #         raise
        
        
    # # WORKS !!!
    # def concatenate_segments(self, background_audio_path, split_video_path) -> str:
    #     """Concatenate all segments into final video with fade transitions, background music and split screen"""
    #     if not self.segments:
    #         raise VideoProcessingError("No segments to concatenate")

    #     clips = []
    #     try:
    #         background_audio = None
    #         if os.path.exists(background_audio_path):
    #             logger.info(f"Loading background audio from: {background_audio_path}")
    #             try:
    #                 background_audio = AudioFileClip(background_audio_path)
    #                 logger.info(f"Successfully loaded background audio with duration: {background_audio.duration}s")
    #             except Exception as e:
    #                 logger.error(f"Error loading background audio: {str(e)}")
    #                 background_audio = None
    #         else:
    #             logger.warning(f"Background audio file not found at {background_audio_path}")

    #         for i, path in enumerate(self.segments):
    #             clip = VideoFileClip(path)
                
    #             if i == 0:
    #                 clip = clip.with_effects([
    #                     vfx.FadeOut(self.transition_duration)
    #                 ])
    #             elif i == len(self.segments) - 1:
    #                 clip = clip.with_effects([
    #                     vfx.FadeIn(self.transition_duration)
    #                 ])
    #             else:
    #                 clip = clip.with_effects([
    #                     vfx.FadeIn(self.transition_duration),
    #                     vfx.FadeOut(self.transition_duration)
    #                 ])
    #             clips.append(clip)

    #         generated_video = concatenate_videoclips(
    #             clips,
    #             method="compose"  
    #         )
    #         split_video = None
    #         if os.path.exists(split_video_path):
    #             try:
    #                 logger.info(f"Loading split video from: {split_video_path}")
    #                 split_video = VideoFileClip(split_video_path)
    #                 logger.info(f"Successfully loaded split video with duration: {split_video.duration}s")
                    
    #                 split_video = split_video.subclipped(0, generated_video.duration)
    #                 logger.info(f"Trimmed split video to match generated video duration: {generated_video.duration}s")
                    
                    
    #                 target_width = min(generated_video.size[0], split_video.size[0])
    #                 generated_video_resized = generated_video.resized(width=target_width)
    #                 split_video_resized = split_video.resized(width=target_width)
                    
    #                 logger.info(f"Resized videos to WIDTH: {target_width}px")
                    
    #                 logger.info("Creating split screen video...")
    #                 final_video = clips_array([
    #                     [generated_video_resized],
    #                     [split_video_resized]
    #                 ])
                    
    #                 logger.info(f"Successfully created split screen video with size: {final_video.size}")
    #             except Exception as e:
    #                 logger.error(f"Error creating split screen: {str(e)}")
    #                 logger.exception("Split screen creation failed, using only generated video")
    #                 final_video = generated_video
    #         else:
    #             logger.warning(f"Split video file not found at {split_video_path}, using only generated video")
    #             final_video = generated_video
            
    #         if background_audio is not None:
    #             try:
    #                 logger.info(f"Video duration: {final_video.duration}s, Audio duration: {background_audio.duration}s")
    #                 if background_audio.duration > final_video.duration:
    #                     logger.info(f"Trimming background audio to match video duration ({final_video.duration}s)")
    #                     background_audio = background_audio.subclipped(0, final_video.duration)
    #                     logger.info(f"New background audio duration: {background_audio.duration}s")
    #                 background_audio_adjusted = background_audio.with_effects([afx.MultiplyVolume(0.5)])
    #                 if final_video.audio is not None:
    #                     original_audio = final_video.audio
    #                     mixed_audio = CompositeAudioClip([
    #                         original_audio, 
    #                         background_audio_adjusted 
    #                     ])
    #                 else:
    #                     mixed_audio = background_audio_adjusted
    #                 final_video = final_video.with_audio(mixed_audio)
    #                 logger.info("Background music added to final video")
    #             except Exception as e:
    #                 logger.error(f"Error applying background audio: {str(e)}")
    #                 logger.exception("Detailed stacktrace:")
            
    #         output_path = os.path.join(self.temp_dir, 'final_video.mp4')
            
    #         final_video.write_videofile(
    #             output_path,
    #             fps=30,
    #             codec='libx264',
    #             audio_codec='aac',
    #             remove_temp=True,
    #             threads=4,
    #             preset='medium'
    #         )
            
    #         return output_path

    #     except Exception as e:
    #         logger.error(f"Failed to concatenate segments: {str(e)}")
    #         logger.exception("Detailed stacktrace:")
    #         raise VideoProcessingError(f"Failed to concatenate segments: {e}")
    #     finally:
    #         for clip in clips:
    #             try:
    #                 clip.close()
    #             except Exception:
    #                 logger.warning(f"Failed to close clip: {clip}")
    #         if 'generated_video' in locals():
    #             try:
    #                 generated_video.close()
    #             except Exception:
    #                 logger.warning("Failed to close generated video")
    #         if 'split_video' in locals() and split_video is not None:
    #             try:
    #                 split_video.close()
    #             except Exception:
    #                 logger.warning("Failed to close split video")
    #         if 'final_video' in locals():
    #             try:
    #                 final_video.close()
    #             except Exception:
    #                 logger.warning("Failed to close final video")
    #         if 'background_audio' in locals() and background_audio is not None:
    #             try:
    #                 background_audio.close()
    #             except Exception:
    #                 logger.warning("Failed to close background audio")
                    
    # def concatenate_segments(self, background_audio_path, split_video_path=None) -> str:
    #     """Concatenate all segments with fade transitions and background music, but no split screen"""
    #     if not self.segments:
    #         raise VideoProcessingError("No segments to concatenate")

    #     clips = []
    #     # try:
    #     background_audio = None
    #     if os.path.exists(background_audio_path):
    #         logger.info(f"Loading background audio from: {background_audio_path}")
    #         try:
    #             background_audio = AudioFileClip(background_audio_path)
    #             logger.info(f"Successfully loaded background audio with duration: {background_audio.duration}s")
    #         except Exception as e:
    #             logger.error(f"Error loading background audio: {str(e)}")
    #             background_audio = None
    #     else:
    #         logger.warning(f"Background audio file not found at {background_audio_path}")

    #     for i, path in enumerate(self.segments):
    #         clip = VideoFileClip(path)
            
    #         if i == 0:
    #             clip = clip.with_effects([
    #                 vfx.FadeOut(self.transition_duration)
    #             ])
    #         elif i == len(self.segments) - 1:
    #             clip = clip.with_effects([
    #                 vfx.FadeIn(self.transition_duration)
    #             ])
    #         else:
    #             clip = clip.with_effects([
    #                 vfx.FadeIn(self.transition_duration),
    #                 vfx.FadeOut(self.transition_duration)
    #             ])
    #         clips.append(clip)

    #     # Simply concatenate the clips without the split screen
    #     final_video = concatenate_videoclips(
    #         clips,
    #         method="compose"  
    #     )
        
    #     # Skip the split screen video code and just use the generated video
        
    #     if background_audio is not None:
    #         try:
    #             logger.info(f"Video duration: {final_video.duration}s, Audio duration: {background_audio.duration}s")
    #             if background_audio.duration > final_video.duration:
    #                 logger.info(f"Trimming background audio to match video duration ({final_video.duration}s)")
    #                 background_audio = background_audio.subclipped(0, final_video.duration)
    #                 logger.info(f"New background audio duration: {background_audio.duration}s")
    #             background_audio_adjusted = background_audio.with_effects([afx.MultiplyVolume(0.5)])
    #             if final_video.audio is not None:
    #                 original_audio = final_video.audio
    #                 mixed_audio = CompositeAudioClip([
    #                     original_audio, 
    #                     background_audio_adjusted 
    #                 ])
    #             else:
    #                 mixed_audio = background_audio_adjusted
    #             final_video = final_video.with_audio(mixed_audio)
    #             logger.info("Background music added to final video")
    #         except Exception as e:
    #             logger.error(f"Error applying background audio: {str(e)}")
    #             logger.exception("Detailed stacktrace:")
        
    #     output_path = os.path.join(self.temp_dir, 'final_video.mp4')
        
    #     final_video.write_videofile(
    #         output_path,
    #         fps=30,
    #         codec='libx264',
    #         audio_codec='aac',
    #         remove_temp=True,
    #         threads=4,
    #         preset='medium'
    #     )
        
    #     return output_path
    

    def cleanup(self):
        """Clean up temporary files and directory"""
        try:
            for segment in self.segments:
                if os.path.exists(segment):
                    os.remove(segment)
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
        
        
        
        
     
    # async def create_segment(self, segment: Dict, index: int, whisper_url: Optional[str] = None, 
    #                        session: Optional[aiohttp.ClientSession] = None) -> str:
    #     """Create a video segment with dynamically synchronized subtitles"""
    #     logger.info(f"Creating segment {index} with Whisper URL: {whisper_url}")
    #     print(f"\n=== Starting Segment {index} Creation ===")
    #     print(f"Whisper URL provided: {whisper_url}")
    #     final_clip = None
        
    #     if not whisper_url:
    #         logger.error("No Whisper URL provided")
    #         print("Error: Missing Whisper URL")
    #         raise VideoProcessingError("Whisper URL is required for subtitle generation")
        
    #     if not session:
    #         logger.error("No session provided")
    #         print("Error: Session is required")
    #         raise VideoProcessingError("Session is required for subtitle generation")
            
    #     try:
    #         print(f"Segment {index} data contains:")
    #         print(f"- Audio data length: {len(segment['audio_data']) if 'audio_data' in segment else 'Missing'}")
    #         print(f"- Image data length: {len(segment['image_data']) if 'image_data' in segment else 'Missing'}")
    #         print(f"- Story text length: {len(segment['story_text']) if 'story_text' in segment else 'Missing'}")
            
    #         audio_path = self._save_base64_audio(segment['audio_data'], index)
    #         print(f"Audio saved to: {audio_path}")
            
    #         image_array = self._decode_base64_image(segment['image_data'])
    #         print("Image decoded successfully")
            
    #         print("\nCreating video clips...")
    #         with AudioFileClip(audio_path) as audio_clip:
    #             duration = audio_clip.duration
    #             print(f"Audio duration: {duration} seconds")
                
    #             video_clip = ImageClip(image_array).with_duration(duration)
    #             video_with_audio = video_clip.with_audio(audio_clip)
                
    #             try:
    #                 print(f"\nAttempting to get subtitles from Whisper API...")
    #                 logger.info(f"Whisper URL provided: {whisper_url}")
                    
    #                 whisper_data = await self.get_synchronized_subtitles(
    #                     segment['audio_data'],
    #                     whisper_url,
    #                     session  
    #                 )
    #                 print("Successfully received whisper data")
                    
    #                 if whisper_data:
    #                     print("Creating subtitle clips...")
    #                     subtitle_clips = self.create_word_level_subtitles(
    #                         whisper_data,
    #                         video_clip.size,
    #                         duration
    #                     )
                        
    #                     if subtitle_clips:
    #                         print(f"Created {len(subtitle_clips)} subtitle clips")
    #                         final_clip = CompositeVideoClip([
    #                             video_with_audio,
    #                             *subtitle_clips
    #                         ])
    #                         print("Composite video created with subtitles")
    #                     else:
    #                         print("No subtitle clips were created, falling back to video without subtitles")
    #                         final_clip = video_with_audio
    #                 else:
    #                     print("No whisper data received, creating video without subtitles")
    #                     final_clip = video_with_audio
                        
    #             except Exception as e:
    #                 logger.error(f"Subtitle generation failed: {str(e)}")
    #                 print(f"Failed to create subtitles: {str(e)}")
    #                 final_clip = video_with_audio
                
    #             output_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
    #             print(f"\nWriting video to: {output_path}")
                
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
    #             print(f"Segment {index} completed successfully")
    #             return output_path
                
    #     except Exception as e:
    #         logger.error(f"Segment {index} creation failed: {str(e)}")
    #         print(f"Error creating segment {index}: {str(e)}")
    #         raise VideoProcessingError(f"Failed to create segment {index}: {str(e)}")
            
    #     finally:
    #         print(f"=== Finishing Segment {index} Creation ===\n")
    #         if final_clip:
    #             try:
    #                 final_clip.close()
    #                 print(f"Cleaned up resources for segment {index}")
    #             except:
    #                 print(f"Warning: Could not clean up resources for segment {index}")
    
    
    
    
    
    # def create_word_level_subtitles(self, whisper_data: Dict, frame_size: tuple, duration: float) -> List:
    #     """Creates word-level subtitle clips"""
    #     try:
    #         subtitle_clips = []
    #         words_data = whisper_data.get('word_level', [])
    #         position = ('center', 0.78)
    #         relative = True
            
    #         for word_data in words_data:
    #             word = word_data['word'].strip()
    #             if not word:
    #                 continue
                    
    #             word_clip = (TextClip(
    #                 text=word,
    #                 font=self.font_path,
    #                 font_size=int(frame_size[1] * 0.075), 
    #                 color='yellow',
    #                 stroke_color='black',
    #                 stroke_width=2
    #             )
    #             .with_position(position, relative=relative)
    #             .with_start(word_data['start'])
    #             .with_duration(word_data['end'] - word_data['start']))
                
    #             subtitle_clips.append(word_clip)
                
    #         return subtitle_clips
    #     except Exception as e:
    #         logger.error(f"Error creating word-level subtitles: {e}")
    #         return []