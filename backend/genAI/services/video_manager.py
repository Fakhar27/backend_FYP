import os
import base64
import tempfile
import logging
from typing import List, Optional
import numpy as np
from PIL import Image
import io
from moviepy import (
    VideoFileClip, ImageClip, AudioFileClip, TextClip,
    concatenate_videoclips, CompositeVideoClip, VideoClip, vfx
)

logger = logging.getLogger(__name__)

class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

class VideoManager:
    def __init__(self):
        """Initialize video manager with temporary directory"""
        try:
            self.temp_dir = tempfile.mkdtemp()
            self.segments: List[str] = []
            self.font_path = self._get_system_font()
            logger.info(f"VideoManager initialized with temp directory: {self.temp_dir}")
        except Exception as e:
            raise VideoProcessingError(f"Failed to initialize VideoManager: {str(e)}")

    def _get_system_font(self) -> str:
        """Get a system font path based on OS"""
        try:
            if os.name == 'nt':  # Windows
                font_paths = [
                    r"C:\Windows\Fonts\arial.ttf",
                    r"C:\Windows\Fonts\calibri.ttf",
                    r"C:\Windows\Fonts\segoeui.ttf"
                ]
            else:  # Linux/Unix
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/TTF/arial.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
                ]

            for path in font_paths:
                if os.path.exists(path):
                    logger.info(f"Using system font: {path}")
                    return path

            raise VideoProcessingError("No suitable system font found")
        except Exception as e:
            logger.error(f"Error finding system font: {str(e)}")
            raise VideoProcessingError(f"Failed to find system font: {str(e)}")

    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Convert base64 image to numpy array"""
        try:
            if 'base64,' in base64_str:
                base64_str = base64_str.split('base64,')[1]
            
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            logger.debug("Successfully decoded base64 image")
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            raise VideoProcessingError(f"Failed to decode base64 image: {str(e)}")

    def _save_base64_audio(self, base64_str: str, index: int) -> str:
        """Save base64 audio to temporary WAV file"""
        try:
            if 'base64,' in base64_str:
                base64_str = base64_str.split('base64,')[1]
                
            audio_data = base64.b64decode(base64_str)
            audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
            
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            
            logger.debug(f"Successfully saved audio to {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Error saving base64 audio: {str(e)}")
            raise VideoProcessingError(f"Failed to save base64 audio: {str(e)}")

    def create_segment(self, segment: dict, index: int) -> str:
        """Create a video segment from image, audio and text"""
        logger.info(f"Creating segment {index}")
        
        audio_clip = None
        video_clip = None
        text_clip = None
        composite_clip = None
        
        try:
            # Step 1: Process Audio
            logger.debug("Processing audio...")
            audio_path = self._save_base64_audio(segment['audio_data'], index)
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            
            # Step 2: Process Image and create video
            logger.debug("Processing image...")
            image_array = self._decode_base64_image(segment['image_data'])
            video_clip = VideoClip(lambda t: image_array)
            video_clip.duration = duration
            video_clip.audio = audio_clip
            
            # Step 3: Create text overlay if story text is provided
            if 'story_text' in segment and segment['story_text']:
                logger.debug("Creating text overlay...")
                text_clip = TextClip(
                    text=segment['story_text'],
                    font=self.font_path,
                    font_size=40,
                    size=(800, None),
                    method='caption',
                    color='white',
                    bg_color=(0, 0, 0, 128),
                    text_align='center',
                    margin=(20, 20),
                    transparent=True,
                    duration=duration
                )
                
                # Combine video and text
                composite_clip = CompositeVideoClip([
                    video_clip,
                    # text_clip.set_position(('center', 'bottom'))
                ])
                composite_clip.duration = duration
                composite_clip.audio = audio_clip
                
                # Add CrossFadeIn effect if not the first clip
                if index > 0:
                    composite_clip = composite_clip.with_effects([vfx.CrossFadeIn(1.0)])
                
                # Save segment with composite
                segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
                logger.info(f"Writing segment to {segment_path}")
                
                composite_clip.write_videofile(
                    segment_path,
                    fps=24,
                    codec='libx264',
                    audio_codec='aac',
                    remove_temp=True,
                    logger=None
                )
            else:
                # Add CrossFadeIn effect if not the first clip
                if index > 0:
                    video_clip = video_clip.with_effects([vfx.CrossFadeIn(1.0)])
                
                # Save segment without text overlay
                segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
                logger.info(f"Writing segment to {segment_path}")
                
                video_clip.write_videofile(
                    segment_path,
                    fps=24,
                    codec='libx264',
                    audio_codec='aac',
                    remove_temp=True,
                    logger=None
                )
            
            self.segments.append(segment_path)
            logger.info(f"Successfully created segment {index}")
            return segment_path
            
        except Exception as e:
            logger.error(f"Error creating segment {index}: {str(e)}")
            raise VideoProcessingError(f"Failed to create segment {index}: {str(e)}")
            
        finally:
            # Cleanup resources
            for clip in [audio_clip, video_clip, text_clip, composite_clip]:
                if clip is not None:
                    try:
                        clip.close()
                    except Exception as e:
                        logger.error(f"Error closing clip: {str(e)}")

    def concatenate_segments(self) -> str:
        """Concatenate segments"""
        logger.info("Starting segment concatenation")
        
        if not self.segments:
            msg = "No segments to concatenate"
            logger.error(msg)
            raise VideoProcessingError(msg)
        
        clips = []
        try:
            # Load all segment clips
            logger.debug("Loading segments...")
            clips = [VideoFileClip(path) for path in self.segments]
            
            # Concatenate segments
            logger.debug("Concatenating segments...")
            final_video = concatenate_videoclips(clips, method="compose")
            
            # Save final video
            output_path = os.path.join(self.temp_dir, 'final_video.mp4')
            logger.info(f"Writing final video to {output_path}")
            
            final_video.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                remove_temp=True,
                logger=None
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error concatenating segments: {str(e)}")
            raise VideoProcessingError(f"Failed to concatenate segments: {str(e)}")
            
        finally:
            # Cleanup clips
            for clip in clips:
                try:
                    clip.close()
                except Exception as e:
                    logger.error(f"Error closing clip: {str(e)}")

    def cleanup(self):
        """Clean up temporary files and directory"""
        logger.info("Starting cleanup")
        
        try:
            # Remove segment files
            for segment in self.segments:
                try:
                    if os.path.exists(segment):
                        os.remove(segment)
                        logger.debug(f"Removed segment: {segment}")
                except Exception as e:
                    logger.error(f"Error removing segment {segment}: {str(e)}")
            
            # Remove temp directory and its contents
            if os.path.exists(self.temp_dir):
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    logger.info(f"Removed temporary directory: {self.temp_dir}")
                except Exception as e:
                    logger.error(f"Error removing temporary directory: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")









# KIND OF WORKS EXCEPT FADEIN PART AND TRANSITION PART
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
