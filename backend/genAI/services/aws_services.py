import boto3
import os
import logging
import tempfile
from botocore.exceptions import ClientError
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

class S3Handler:
    """Handler for AWS S3 operations"""
    
    def __init__(self):
        """Initialize S3 client with credentials from environment variables"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_S3_REGION_NAME')
        )
        self.bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')
        self.temp_dir = tempfile.mkdtemp()
        
        self.background_videos = {
            "urban": "background_videos/split_screen_video_1.mp4",
            "1": "background_videos/split_screen_video_1.mp4",
            "2":" background_videos/nature.mp4",
            "3": "background_videos/subway.mp4",
        }
        
        self.background_music = {
            "synthwave": "background_music/backgroundMusic1.wav",
            "1": "background_music/backgroundMusic1.wav",
            "2": "background_music/audio2.wav",
            "3": "background_music/Aadat.wav",
        }
        
        logger.info(f"S3Handler initialized with bucket: {self.bucket_name}")
    
    # def get_media_file(self, media_type: str, selection: str) -> Optional[str]:
    #     """
    #     Download a media file from S3 based on type and selection
        
    #     Args:
    #         media_type: 'video' or 'music'
    #         selection: Key from the mapping dictionaries
            
    #     Returns:
    #         Local path to downloaded file or None if error
    #     """
    #     try:
    #         if media_type == 'video':
    #             mapping = self.background_videos
    #             default_key = list(mapping.values())[0]
    #         elif media_type == 'music':
    #             mapping = self.background_music
    #             default_key = list(mapping.values())[0]
    #         else:
    #             logger.error(f"Invalid media type: {media_type}")
    #             return None
                
    #         s3_key = mapping.get(selection, default_key)
    #         filename = s3_key.split('/')[-1]
    #         local_path = os.path.join(self.temp_dir, filename)
            
    #         logger.info(f"Downloading {s3_key} from bucket {self.bucket_name}")
            
    #         self.s3_client.download_file(
    #             self.bucket_name,
    #             s3_key,
    #             local_path
    #         )
            
    #         logger.info(f"Successfully downloaded to {local_path}")
    #         return local_path
            
    #     except ClientError as e:
    #         logger.error(f"Error downloading from S3: {str(e)}")
    #         # if media_type == 'video':
    #         #     return "E:\\fyp_backend\\backend\\genAI\\split_screen_video_1.mp4"
    #         # else:
    #         #     return "E:\\fyp_backend\\backend\\genAI\\backgroundMusic1.wav"
    #     except Exception as e:
    #         logger.error(f"Unexpected error: {str(e)}")
    #         return None
    def get_media_file(self, media_type: str, selection: str) -> Optional[str]:
        """
        Download a media file from S3 based on type and selection
        
        Args:
            media_type: 'video' or 'music'
            selection: Key from the mapping dictionaries
            
        Returns:
            Local path to downloaded file, None if error, or special "NONE" string
            if user selected "0" (no background)
        """
        try:
            # Special case: Check if the selection is "0" (no background)
            if selection == "0":
                logger.info(f"User selected no {media_type} background")
                return "NONE"  # Return special marker for "no background"
                
            if media_type == 'video':
                mapping = self.background_videos
                default_key = list(mapping.values())[0]
            elif media_type == 'music':
                mapping = self.background_music
                default_key = list(mapping.values())[0]
            else:
                logger.error(f"Invalid media type: {media_type}")
                return None
                
            s3_key = mapping.get(selection, default_key)
            filename = s3_key.split('/')[-1]
            local_path = os.path.join(self.temp_dir, filename)
            
            logger.info(f"Downloading {s3_key} from bucket {self.bucket_name}")
            
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                local_path
            )
            
            logger.info(f"Successfully downloaded to {local_path}")
            return local_path
            
        except ClientError as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None
    
    def list_available_media(self, media_type: str) -> List[str]:
        """
        List available media options for a given type
        
        Args:
            media_type: 'video' or 'music'
            
        Returns:
            List of available selection keys
        """
        if media_type == 'video':
            return list(self.background_videos.keys())
        elif media_type == 'music':
            return list(self.background_music.keys())
        else:
            return []
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")