import os
import base64
import tempfile
import logging
from typing import List, Optional
import numpy as np
from PIL import Image
import io
from moviepy import VideoFileClip,ImageClip,AudioFileClip,concatenate_videoclips,CompositeVideoClip,VideoClip
# from moviepy.video.VideoClip import ImageClip
# from moviepy.audio.io.AudioFileClip import AudioFileClip
# from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
# from moviepy.video.compositing.concatenate import concatenate_videoclips

logger = logging.getLogger(__name__)

class VideoManager:
    def __init__(self):
        """Initialize video manager with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.segments: List[str] = []  # Store paths to segment files
        logger.info(f"VideoManager initialized with temp directory: {self.temp_dir}")

    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Convert base64 image to numpy array"""
        try:
            # Remove data URL prefix if present
            if 'base64,' in base64_str:
                base64_str = base64_str.split('base64,')[1]
            
            # Decode base64 to image
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            logger.debug("Successfully decoded base64 image")
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            raise

    def _save_base64_audio(self, base64_str: str, index: int) -> str:
        """Save base64 audio to temporary WAV file"""
        try:
            if 'base64,' in base64_str:
                base64_str = base64_str.split('base64,')[1]
                
            # Decode and save audio
            audio_data = base64.b64decode(base64_str)
            audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
            
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            
            logger.debug(f"Successfully saved audio to {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Error saving base64 audio: {str(e)}")
            raise

    def create_segment(self, segment: dict, index: int) -> str:
        """Create a video segment from image and audio"""
        logger.info(f"Creating segment {index}")
        
        audio_clip = None
        image_clip = None
        
        try:
            # Step 1: Process Audio
            logger.debug("Processing audio...")
            audio_path = self._save_base64_audio(segment['audio_data'], index)
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            
            # Step 2: Process Image and create video
            logger.debug("Processing image...")
            image_array = self._decode_base64_image(segment['image_data'])
            # Create ImageClip with duration
            video_clip = VideoClip(lambda t: image_array)
            video_clip.duration = duration
            video_clip.audio = audio_clip
            
            # Step 3: Save segment
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
            raise
            
        finally:
            # Cleanup resources
            if audio_clip:
                try:
                    audio_clip.close()
                except Exception as e:
                    logger.error(f"Error closing audio clip: {str(e)}")
            if video_clip:
                try:
                    video_clip.close()
                except Exception as e:
                    logger.error(f"Error closing video clip: {str(e)}")

    def concatenate_segments(self) -> str:
        """Concatenate all segments into final video"""
        logger.info("Starting segment concatenation")
        
        if not self.segments:
            msg = "No segments to concatenate"
            logger.error(msg)
            raise ValueError(msg)
        
        clips = []
        try:
            # Load all segment clips
            logger.debug("Loading segments...")
            clips = [VideoFileClip(path) for path in self.segments]
            
            # Concatenate
            logger.debug("Concatenating segments...")
            final_video = concatenate_videoclips(clips, method="chain")
            
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
            raise
            
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
            
            # Remove temp directory
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
                logger.info(f"Removed temporary directory: {self.temp_dir}")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


















# from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip
# from moviepy.video.tools.drawing import color_gradient
# import tempfile
# from moviepy.video.VideoClip import ColorClip
# import os
# import base64
# import logging
# from PIL import Image
# import io
# import numpy as np
# from typing import List, Optional
# from pydantic import BaseModel

# logger = logging.getLogger(__name__)

# class VideoSegment(BaseModel):
#     """Model for video segment data"""
#     image_data: str  # base64
#     audio_data: str  # base64
#     story_text: str
#     duration: Optional[float] = None

# class VideoManager:
#     def __init__(self):
#         """Initialize video manager with temporary directory"""
#         self.temp_dir = tempfile.mkdtemp()
#         logger.info(f"Created temporary directory: {self.temp_dir}")
#         self.segments: List[str] = []  # Store paths to segment files

#     def _decode_base64_image(self, base64_str: str) -> np.ndarray:
#         """Convert base64 image to numpy array"""
#         try:
#             # Remove data URL prefix if present
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
            
#             image_data = base64.b64decode(base64_str)
#             image = Image.open(io.BytesIO(image_data))
#             # Convert to RGB if necessary
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
#             return np.array(image)
#         except Exception as e:
#             logger.error(f"Error decoding image: {str(e)}")
#             raise

#     def _save_base64_audio(self, base64_str: str, index: int) -> str:
#         """Save base64 audio to temporary WAV file"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
                
#             audio_data = base64.b64decode(base64_str)
#             audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
            
#             with open(audio_path, 'wb') as f:
#                 f.write(audio_data)
            
#             return audio_path
#         except Exception as e:
#             logger.error(f"Error saving audio: {str(e)}")
#             raise

#     def create_text_clip(self, text: str, size: tuple, duration: float) -> TextClip:
#         """Create a text clip with proper styling"""
#         try:
#             # Calculate text size based on image width
#             fontsize = int(size[0] * 0.04)  # 4% of image width
            
#             # Create text clip with proper configuration
#             txt_clip = TextClip(
#                 text,
#                 font='Arial',
#                 fontsize=fontsize,
#                 color='white',
#                 size=(size[0], None),
#                 method='caption',
#                 align='center',
#                 stroke_color='black',
#                 stroke_width=2
#             )
            
#             # Create semi-transparent background
#             txt_bg = ColorClip(
#                 size=(size[0], txt_clip.h + 20),
#                 color=(0, 0, 0)
#             ).set_opacity(0.6)
            
#             # Combine text and background
#             return (CompositeVideoClip([
#                 txt_bg,
#                 txt_clip.set_position('center')
#             ])
#             .set_position(('center', 'bottom'))
#             .set_duration(duration))
            
#         except Exception as e:
#             logger.error(f"Error creating text clip: {str(e)}")
#             raise

#     def create_segment(self, segment: VideoSegment, index: int) -> str:
#         """Create a video segment from image and audio with text overlay"""
#         try:
#             logger.info(f"Creating segment {index}")
            
#             # Convert base64 image to numpy array
#             image_array = self._decode_base64_image(segment.image_data)
            
#             # Save audio to temporary file
#             audio_path = self._save_base64_audio(segment.audio_data, index)
            
#             # Create audio clip and get its duration
#             audio_clip = AudioFileClip(audio_path)
#             duration = audio_clip.duration
            
#             # Create image clip with duration
#             image_clip = ImageClip(image_array).duration(duration)
            
#             # Get image size for text clip
#             img_size = image_clip.size
            
#             # Create text clip
#             text_clip = self.create_text_clip(
#                 segment.story_text,
#                 img_size,
#                 duration
#             )
            
#             # Combine image, text, and audio
#             video = (CompositeVideoClip([
#                 image_clip,
#                 text_clip
#             ])
#             .set_audio(audio_clip))
            
#             # Save segment
#             segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#             video.write_videofile(
#                 segment_path,
#                 fps=24,
#                 codec='libx264',
#                 audio_codec='aac',
#                 temp_audiofile=os.path.join(self.temp_dir, f'temp_audio_{index}.m4a'),
#                 remove_temp=True,
#                 logger=None  # Suppress moviepy progress bars
#             )
            
#             # Close clips to free up resources
#             video.close()
#             audio_clip.close()
#             image_clip.close()
#             text_clip.close()
            
#             self.segments.append(segment_path)
#             logger.info(f"Successfully created segment {index}")
#             return segment_path
            
#         except Exception as e:
#             logger.error(f"Error creating segment {index}: {str(e)}")
#             raise

#     def concatenate_segments(self) -> str:
#         """Concatenate all segments into final video"""
#         try:
#             if not self.segments:
#                 raise ValueError("No segments to concatenate")
            
#             logger.info("Loading video clips for concatenation")
#             clips = []
#             for path in self.segments:
#                 try:
#                     clip = VideoFileClip(path)
#                     clips.append(clip)
#                 except Exception as e:
#                     logger.error(f"Error loading clip {path}: {str(e)}")
#                     raise
            
#             logger.info("Concatenating video clips")
#             final_video = concatenate_videoclips(clips, method="compose")
            
#             # Save final video
#             output_path = os.path.join(self.temp_dir, 'final_video.mp4')
#             logger.info(f"Writing final video to {output_path}")
            
#             final_video.write_videofile(
#                 output_path,
#                 fps=24,
#                 codec='libx264',
#                 audio_codec='aac',
#                 temp_audiofile=os.path.join(self.temp_dir, 'temp_audio_final.m4a'),
#                 remove_temp=True,
#                 logger=None  # Suppress moviepy progress bars
#             )
            
#             # Cleanup clips
#             for clip in clips:
#                 clip.close()
#             final_video.close()
            
#             return output_path
            
#         except Exception as e:
#             logger.error(f"Error concatenating segments: {str(e)}")
#             raise

#     def cleanup(self):
#         """Clean up temporary files and directory"""
#         try:
#             logger.info("Starting cleanup")
            
#             # First, try to close any active file handles
#             import gc
#             gc.collect()
            
#             # Wait a bit to ensure files are released
#             import time
#             time.sleep(1)
            
#             for segment in self.segments:
#                 try:
#                     if os.path.exists(segment):
#                         os.remove(segment)
#                         logger.info(f"Removed segment: {segment}")
#                 except Exception as e:
#                     logger.error(f"Error removing segment {segment}: {str(e)}")
            
#             # Remove any remaining files in temp directory
#             for filename in os.listdir(self.temp_dir):
#                 try:
#                     file_path = os.path.join(self.temp_dir, filename)
#                     if os.path.isfile(file_path):
#                         os.unlink(file_path)
#                         logger.info(f"Removed file: {file_path}")
#                 except Exception as e:
#                     logger.error(f"Error removing file {filename}: {str(e)}")
            
#             try:
#                 os.rmdir(self.temp_dir)
#                 logger.info(f"Removed temporary directory: {self.temp_dir}")
#             except Exception as e:
#                 logger.error(f"Error removing temporary directory: {str(e)}")
                
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")








# from moviepy.video.VideoClip import ColorClip
# import tempfile
# import os
# import base64
# import logging
# from PIL import Image
# import io
# import numpy as np
# from typing import List, Optional
# from pydantic import BaseModel

# logger = logging.getLogger(__name__)

# class VideoSegment(BaseModel):
#     """Model for video segment data"""
#     image_data: str  # base64
#     audio_data: str  # base64
#     story_text: str
#     duration: Optional[float] = None

# class VideoManager:
#     def __init__(self):
#         """Initialize video manager with temporary directory"""
#         self.temp_dir = tempfile.mkdtemp()
#         logger.info(f"Created temporary directory: {self.temp_dir}")
#         self.segments: List[str] = []  # Store paths to segment files

#     def _decode_base64_image(self, base64_str: str) -> np.ndarray:
#         """Convert base64 image to numpy array"""
#         try:
#             # Remove data URL prefix if present
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
            
#             image_data = base64.b64decode(base64_str)
#             image = Image.open(io.BytesIO(image_data))
#             # Convert to RGB if necessary
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
#             return np.array(image)
#         except Exception as e:
#             logger.error(f"Error decoding image: {str(e)}")
#             raise

#     def _save_base64_audio(self, base64_str: str, index: int) -> str:
#         """Save base64 audio to temporary WAV file"""
#         try:
#             if 'base64,' in base64_str:
#                 base64_str = base64_str.split('base64,')[1]
                
#             audio_data = base64.b64decode(base64_str)
#             audio_path = os.path.join(self.temp_dir, f'audio_{index}.wav')
            
#             with open(audio_path, 'wb') as f:
#                 f.write(audio_data)
            
#             return audio_path
#         except Exception as e:
#             logger.error(f"Error saving audio: {str(e)}")
#             raise

#     def create_segment(self, segment: VideoSegment, index: int) -> str:
#         """Create a video segment from image and audio with text overlay"""
#         try:
#             logger.info(f"Creating segment {index}")
            
#             # Convert base64 image to numpy array
#             image_array = self._decode_base64_image(segment.image_data)
            
#             # Save audio to temporary file
#             audio_path = self._save_base64_audio(segment.audio_data, index)
            
#             # Create audio clip and get its duration
#             audio_clip = AudioFileClip(audio_path)
#             duration = audio_clip.duration
            
#             # Create image clip with duration
#             image_clip = ImageClip(image_array)
#             image_clip = image_clip.duration(duration)  # This is the fixed line
            
#             # Create text clip
#             text_clip = (TextClip(
#                 segment.story_text,
#                 fontsize=30,
#                 color='white',
#                 bg_color='rgba(0,0,0,0.5)',
#                 font='Arial',
#                 size=(image_clip.size[0], None),  # Use image size for width
#                 method='caption'
#             )
#             .set_position('bottom')
#             .set_duration(duration))
            
#             # Combine image, text, and audio
#             video = (CompositeVideoClip([image_clip, text_clip])
#                     .set_audio(audio_clip))
            
#             # Save segment
#             segment_path = os.path.join(self.temp_dir, f'segment_{index}.mp4')
#             video.write_videofile(
#                 segment_path,
#                 fps=24,
#                 codec='libx264',
#                 audio_codec='aac',
#                 temp_audiofile=os.path.join(self.temp_dir, f'temp_audio_{index}.m4a'),
#                 remove_temp=True
#             )
            
#             # Cleanup clips
#             video.close()
#             audio_clip.close()
            
#             self.segments.append(segment_path)
#             logger.info(f"Successfully created segment {index}")
#             return segment_path
            
#         except Exception as e:
#             logger.error(f"Error creating segment {index}: {str(e)}")
#             raise

#     def concatenate_segments(self) -> str:
#         """Concatenate all segments into final video"""
#         try:
#             if not self.segments:
#                 raise ValueError("No segments to concatenate")
            
#             logger.info("Loading video clips for concatenation")
#             clips = [VideoFileClip(path) for path in self.segments]
            
#             logger.info("Concatenating video clips")
#             final_video = concatenate_videoclips(clips)
            
#             # Save final video
#             output_path = os.path.join(self.temp_dir, 'final_video.mp4')
#             logger.info(f"Writing final video to {output_path}")
            
#             final_video.write_videofile(
#                 output_path,
#                 fps=24,
#                 codec='libx264',
#                 audio_codec='aac',
#                 temp_audiofile=os.path.join(self.temp_dir, 'temp_audio_final.m4a'),
#                 remove_temp=True
#             )
            
#             # Cleanup clips
#             for clip in clips:
#                 clip.close()
#             final_video.close()
            
#             return output_path
            
#         except Exception as e:
#             logger.error(f"Error concatenating segments: {str(e)}")
#             raise

#     def cleanup(self):
#         """Clean up temporary files and directory"""
#         try:
#             logger.info("Starting cleanup")
#             for segment in self.segments:
#                 try:
#                     if os.path.exists(segment):
#                         os.remove(segment)
#                         logger.info(f"Removed segment: {segment}")
#                 except Exception as e:
#                     logger.error(f"Error removing segment {segment}: {str(e)}")
            
#             # Remove any remaining files in temp directory
#             for filename in os.listdir(self.temp_dir):
#                 try:
#                     file_path = os.path.join(self.temp_dir, filename)
#                     if os.path.isfile(file_path):
#                         os.unlink(file_path)
#                         logger.info(f"Removed file: {file_path}")
#                 except Exception as e:
#                     logger.error(f"Error removing file {filename}: {str(e)}")
            
#             try:
#                 os.rmdir(self.temp_dir)
#                 logger.info(f"Removed temporary directory: {self.temp_dir}")
#             except Exception as e:
#                 logger.error(f"Error removing temporary directory: {str(e)}")
                
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")