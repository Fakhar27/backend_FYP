from typing import List, Dict, Any, Optional
import logging
import aiohttp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

logger = logging.getLogger(__name__)

class ContentRequest(BaseModel):
    prompt: str = Field(..., description="User's content prompt")
    genre: str = Field(..., description="Content category/genre")
    iterations: int = Field(default=4, ge=1, le=10)

class ContentResponse(BaseModel):
    story: str
    enhanced_story: str
    image_url: Optional[str]
    iteration: int

class LangChainService:
    def __init__(
        self,
        openai_api_key: str,
        colab_url: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
        # Story LLM with strict token limit
        self.story_llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=openai_api_key,
            max_tokens=25  # Strict limit for one-sentence stories
        )
        
        # Image prompt LLM without token limit
        self.image_llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=openai_api_key,
            max_tokens=200  # Allow detailed image prompts
        )
        
        self.colab_url = colab_url
        self._session = None
        self._session_refs = 0
        
        # Story generation prompt with strict length control and continuity
        self.story_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             """You are a cinematic storyteller creating a suspenseful narrative.
             Generate EXACTLY ONE SENTENCE (max 15-20 words) that:
             1. Continues directly from the previous scene
             2. Focuses on one clear, dramatic moment or action
             3. Uses vivid, visual language
             4. Maintains tension and atmosphere
             
             STRICT RULES:
             - ONE sentence only
             - Maximum 20 words
             - Must be immediately filmable
             - Focus on visual action and mood"""),
            ("human", 
             """Create the next scene in this {genre} story about: {prompt}
             Previous scene: {context}
             
             Remember:
             - ONE dramatic sentence
             - Direct continuation
             - Clear visual action
             - Maximum 20 words""")
        ])
        
        # Detailed image prompt template
        self.image_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             """You are a master of cinematic image composition. Create detailed Stable Diffusion prompts that:
             1. Maintain perfect visual continuity with previous scenes
             2. Specify exact lighting, colors, and atmosphere
             3. Include technical camera details and composition
             4. Describe environment with rich detail
             
             Critical elements to maintain:
             - Character appearance and positioning
             - Time of day and weather conditions
             - Color palette and lighting style
             - Environmental details and props
             - Camera angle and framing
             
             Use format: [Scene Description], [Atmosphere], [Technical Details], [Style Keywords]"""),
            ("human", 
             """Create a highly detailed image prompt that maintains perfect visual continuity.
             
             Story context: {context}
             Previous scene details: {prev_context}
             
             Required elements:
             1. Precise character and prop descriptions
             2. Exact lighting and atmosphere details
             3. Specific camera angles and framing
             4. Style keywords (cinematic, dramatic, professional)
             5. Technical specifications for consistent look
             
             Make the description detailed enough for perfect visual matching.""")
        ])

    async def session(self):
        """Async property to manage session lifecycle"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_refs = 0
        self._session_refs += 1
        return self._session

    async def _release_session(self):
        """Safely release session reference"""
        if self._session is None:
            return
            
        self._session_refs -= 1
        if self._session_refs <= 0 and not self._session.closed:
            await self._session.close()
            self._session = None

    async def generate_story(self, prompt: str, genre: str, context: str = "") -> str:
        """Generate concise, one-sentence story"""
        try:
            story_chain = self.story_prompt | self.story_llm | StrOutputParser()
            response = await story_chain.ainvoke({
                "genre": genre,
                "prompt": prompt,
                "context": context or "Initial scene"
            })
            logger.info(f"Generated story: {response}")
            return response
        except Exception as e:
            logger.error(f"Story generation failed: {str(e)}")
            raise

    async def enhance_story(self, context: str, prev_context: str = "") -> str:
        """Convert story into detailed image generation prompt"""
        try:
            image_chain = self.image_prompt | self.image_llm | StrOutputParser()
            response = await image_chain.ainvoke({
                "context": context,
                "prev_context": prev_context
            })
            logger.info(f"Enhanced story for image generation: {response}")
            return response
        except Exception as e:
            logger.error(f"Story enhancement failed: {str(e)}")
            raise

    async def generate_image(self, prompt: str) -> Optional[str]:
        """Generate image with proper session handling"""
        if not self.colab_url:
            logger.error("COLAB_URL not set")
            return None
            
        try:
            session = await self.session()
            logger.info(f"Sending image generation request to: {self.colab_url}/generate-image")
            logger.info(f"With prompt: {prompt}")
            
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
                
                logger.info("Image data successfully received")
                return image_data
                
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return None
        finally:
            await self._release_session()

    async def generate_content_pipeline(
        self,
        request: ContentRequest
    ) -> List[ContentResponse]:
        """Main content generation pipeline with improved coherence"""
        results = []
        context = ""
        prev_enhanced_story = ""

        try:
            for i in range(request.iterations):
                logger.info(f"Starting iteration {i + 1}")
                
                # Generate coherent story continuation
                story = await self.generate_story(
                    prompt=request.prompt,
                    genre=request.genre,
                    context=context
                )
                
                # Create detailed image prompt with visual consistency
                enhanced_story = await self.enhance_story(
                    story,
                    prev_enhanced_story
                )
                
                # Generate matching image
                image_url = await self.generate_image(enhanced_story)
                
                result = ContentResponse(
                    story=story,
                    enhanced_story=enhanced_story,
                    image_url=image_url,
                    iteration=i + 1
                )
                
                results.append(result)
                context = story
                prev_enhanced_story = enhanced_story
                
                logger.info(f"Completed iteration {i + 1}")
                
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            self._session_refs = 0

    def __del__(self):
        """Ensure cleanup on deletion"""
        if self._session and not self._session.closed:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
                else:
                    loop.run_until_complete(self.cleanup())
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")















# from typing import List, Dict, Any, Optional
# import logging
# import aiohttp
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from pydantic import BaseModel, Field
# import json

# logger = logging.getLogger(__name__)

# class ContentRequest(BaseModel):
#     prompt: str = Field(..., description="User's content prompt")
#     genre: str = Field(..., description="Content category/genre")
#     iterations: int = Field(default=4, ge=1, le=10)

# class ContentResponse(BaseModel):
#     story: str
#     enhanced_story: str
#     image_url: Optional[str]
#     iteration: int

# class LangChainService:
#     def __init__(
#         self,
#         openai_api_key: str,
#         colab_url: str,
#         model_name: str = "gpt-4o-mini",
#         max_tokens=50,
#         temperature: float = 0.7
#     ):
#         self.llm = ChatOpenAI(
#             model_name=model_name,
#             temperature=temperature,
#             openai_api_key=openai_api_key,
#             max_tokens=50
#         )
#         self.colab_url = colab_url
#         self.session = None
#         self._session_refs  = 0
        
#         @property
#         async def session(self):
#             """Async property to get or create session"""
#             if self._session is None or self._session.closed:
#                 self._session = aiohttp.ClientSession()
#                 self._session_refs = 0
#             self._session_refs += 1
#             return self._session

#         async def _release_session(self):
#             """Decrease reference count and close session if no more references"""
#             self._session_refs -= 1
#             if self._session_refs <= 0 and self._session and not self._session.closed:
#                 await self._session.close()
#                 self._session = None
        
#         self.base_prompt = ChatPromptTemplate.from_messages([
#             ("system", 
#              """You are a visual storyteller creating a continuous narrative across multiple scenes.
#              For each scene, create EXACTLY ONE SHORT SENTENCE (15-20 words maximum) that:
#              1. Maintains visual and narrative continuity with the previous scene
#              2. Focuses on the main character's journey/progression
#              3. Emphasizes clear, filmable visual elements
#              4. Keeps the same setting, time of day, and atmosphere as previous scenes"""),
#             ("human", 
#              """Create the next scene in a {genre} story about: {prompt}
#              Previous scene (maintain continuity): {context}
#              Requirements:
#              - ONE sentence only (15-20 words max)
#              - Same setting/atmosphere as previous scene
#              - Clear visual progression from previous scene
#              - Focus on immediate, visible action""")
#         ])
        
#         self.enhancement_prompt = ChatPromptTemplate.from_messages([
#             ("system", 
#              """Convert stories into precise image generation prompts that maintain visual consistency.
#              Focus on:
#              1. Matching the lighting, color scheme, and atmosphere of previous scenes
#              2. Keeping the same character appearance and setting details
#              3. Using consistent artistic style keywords
#              4. Emphasizing scene-to-scene continuity"""),
#             ("human", 
#              """Convert this story into a Stable Diffusion prompt. Maintain visual consistency with:
#              Previous context: {prev_context}
#              Current story: {context}
             
#              Requirements:
#              - Match lighting/colors from previous scene
#              - Same character appearance and setting
#              - Include artistic style keywords (cinematic, dynamic composition)
#              - 50 words maximum""")
#         ])

#     async def _ensure_session(self):
#         """Ensure aiohttp session exists and is active"""
#         if self.session is None or self.session.closed:
#             self.session = aiohttp.ClientSession()
#         return self.session

#     async def _cleanup_session(self):
#         """Cleanup session if it exists"""
#         if self.session and not self.session.closed:
#             await self.session.close()
#             self.session = None

#     async def generate_story(self, prompt: str, genre: str, context: str = "") -> str:
#         """Generate story using OpenAI"""
#         try:
#             story_chain = self.base_prompt | self.llm | StrOutputParser()
#             response = await story_chain.ainvoke({
#                 "genre": genre,
#                 "prompt": prompt,
#                 "context": context or "None"
#             })
#             logger.info(f"Generated story: {response}")
#             return response
#         except Exception as e:
#             logger.error(f"Story generation failed: {str(e)}")
#             raise

#     # async def enhance_story(self, context: str) -> str:
#     #     """Enhance story for better visualization"""
#     #     try:
#     #         enhancement_chain = self.enhancement_prompt | self.llm | StrOutputParser()
#     #         response = await enhancement_chain.ainvoke({"context": context})
#     #         logger.info(f"Enhanced story: {response}")
#     #         return response
#     #     except Exception as e:
#     #         logger.error(f"Story enhancement failed: {str(e)}")
#     #         raise
#     async def enhance_story(self, context: str, prev_context: str = "") -> str:
#         """Enhanced version with coherence control"""
#         try:
#             enhancement_chain = self.enhancement_prompt | self.llm | StrOutputParser()
#             response = await enhancement_chain.ainvoke({
#                 "context": context,
#                 "prev_context": prev_context or "Initial scene"
#             })
#             return response
#         except Exception as e:
#             logger.error(f"Story enhancement failed: {str(e)}")
#             raise

#     async def generate_image(self, prompt: str) -> Optional[str]:
#         """Generate image using Flask/AWS endpoint"""
#         if not self.colab_url:
#             logger.error("COLAB_URL not set")
#             return None
            
#         try:
#             session = await self._ensure_session()
#             logger.info(f"Sending image generation request to: {self.colab_url}/generate-image")
#             logger.info(f"With prompt: {prompt}")
            
#             async with session.post(
#                 f"{self.colab_url}/generate-image",
#                 json={"prompt": prompt},
#                 timeout=aiohttp.ClientTimeout(total=60)
#             ) as response:
#                 response.raise_for_status()
#                 result = await response.json()
                
#                 logger.info("Image generation response received")
                
#                 if 'error' in result:
#                     logger.error(f"Error from image generation: {result['error']}")
#                     return None
                    
#                 image_data = result.get('image_data')
#                 if not image_data:
#                     logger.error("No image data in response")
#                     return None
                
#                 # The image data is already a complete data URL
#                 logger.info("Image data successfully received")
#                 return image_data
                
#         except Exception as e:
#             logger.error(f"Image generation failed: {str(e)}")
#             return None

#     async def generate_content_pipeline(
#         self,
#         request: ContentRequest
#     ) -> List[ContentResponse]:
#         results = []
#         context = ""
#         prev_enhanced_story = ""

#         try:
#             for i in range(request.iterations):
#                 story = await self.generate_story(
#                     prompt=request.prompt,
#                     genre=request.genre,
#                     context=context
#                 )
                
#                 enhanced_story = await self.enhance_story(
#                     story,
#                     prev_enhanced_story
#                 )
                
#                 image_url = await self.generate_image(enhanced_story)
                
#                 result = ContentResponse(
#                     story=story,
#                     enhanced_story=enhanced_story,
#                     image_url=image_url,
#                     iteration=i + 1
#                 )
                
#                 results.append(result)
#                 context = story
#                 prev_enhanced_story = enhanced_story
                
#             return results
            
#         except Exception as e:
#             logger.error(f"Pipeline execution failed: {str(e)}")
#             raise

#     def __del__(self):
#         """Cleanup when service is destroyed"""
#         import asyncio
#         if self.session and not self.session.closed:
#             asyncio.run(self._cleanup_session())








# from typing import List, Dict, Any, Optional
# import logging
# import aiohttp
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from pydantic import BaseModel, Field
# import json
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ContentRequest(BaseModel):
#     """Content generation request model"""
#     prompt: str = Field(..., description="User's content prompt")
#     genre: str = Field(..., description="Content category/genre")
#     iterations: int = Field(default=4, ge=1, le=10, description="Number of iterations")

# class ContentResponse(BaseModel):
#     """Content generation response model"""
#     story: str
#     enhanced_story: str
#     image_url: Optional[str]

# class LangChainService:
#     def __init__(
#         self,
#         openai_api_key: str,
#         colab_url: str,
#         model_name: str = "gpt-4o-mini",
#         temperature: float = 0.7
#     ):
#         # Verify API key
#         if not openai_api_key:
#             openai_api_key = os.getenv('OPENAI_API_KEY',"")
#             if not openai_api_key:
#                 raise ValueError("OpenAI API key must be provided either directly or through OPENAI_API_KEY environment variable")
        
#         self.llm = ChatOpenAI(
#             model_name=model_name,
#             temperature=temperature,
#             openai_api_key=openai_api_key
#         )
#         self.colab_url = colab_url
#         self._session = None  # aiohttp session
        
#         # Initialize prompt templates
#         self.base_prompt = ChatPromptTemplate.from_messages([
#             ("system", 
#              """You are a creative content generator specialized in {genre} content. 
#              Your task is to create engaging, visual stories perfect for short-form video content."""),
#             ("human", 
#              """Create a captivating {genre} story based on: {prompt}. 
#              Make it brief (2-3 sentences), highly visual, and engaging. 
#              Focus on elements that would work well in a short video or reel.""")
#         ])
        
#         self.enhancement_prompt = ChatPromptTemplate.from_messages([
#             ("system", 
#              """You are an expert at enhancing visual descriptions for AI image generation.
#              Your task is to add specific visual details that will help generate better images."""),
#             ("human", 
#              """Enhance this story with vivid visual details while maintaining its essence.
#              Focus on visual elements like lighting, colors, composition, and specific details.
#              Story: {context}""")
#         ])

#     async def _get_session(self) -> aiohttp.ClientSession:
#         """Get or create aiohttp session"""
#         if self._session is None or self._session.closed:
#             self._session = aiohttp.ClientSession()
#         return self._session

#     async def generate_story(self, prompt: str, genre: str, context: str = "") -> str:
#         """Generate story using OpenAI"""
#         try:
#             story_chain = self.base_prompt | self.llm | StrOutputParser()
#             full_prompt = prompt if not context else f"{prompt} (Previous context: {context})"
#             return await story_chain.ainvoke({
#                 "genre": genre,
#                 "prompt": full_prompt
#             })
#         except Exception as e:
#             logger.error(f"Story generation failed: {str(e)}")
#             raise

#     async def enhance_story(self, context: str) -> str:
#         """Enhance story for better visualization"""
#         try:
#             enhancement_chain = self.enhancement_prompt | self.llm | StrOutputParser()
#             return await enhancement_chain.ainvoke({"context": context})
#         except Exception as e:
#             logger.error(f"Story enhancement failed: {str(e)}")
#             raise

#     async def generate_image(self, prompt: str) -> Optional[str]:
#         """Generate image using Flask/AWS endpoint"""
#         if not self.colab_url:
#             logger.error("COLAB_URL not set")
#             return None
            
#         try:
#             session = await self._get_session()
#             async with session.post(
#                 f"{self.colab_url}/generate-image",
#                 json={"prompt": prompt},
#                 timeout=aiohttp.ClientTimeout(total=30)
#             ) as response:
#                 response.raise_for_status()
#                 result = await response.json()
#                 return result.get("image_url")
                
#         except Exception as e:
#             logger.error(f"Image generation failed: {str(e)}")
#             return None

#     async def generate_content_iteration(
#         self,
#         prompt: str,
#         genre: str,
#         context: str = ""
#     ) -> ContentResponse:
#         """Generate a single iteration of content"""
#         try:
#             # Generate base story
#             story = await self.generate_story(prompt, genre, context)

#             # Enhance story for better visualization
#             enhanced_story = await self.enhance_story(story)

#             # Generate image
#             image_url = await self.generate_image(enhanced_story)

#             return ContentResponse(
#                 story=story,
#                 enhanced_story=enhanced_story,
#                 image_url=image_url
#             )
#         except Exception as e:
#             logger.error(f"Content generation failed: {str(e)}")
#             raise

#     async def generate_content_pipeline(
#         self,
#         request: ContentRequest
#     ) -> List[ContentResponse]:
#         """Generate full content pipeline with multiple iterations"""
#         results = []
#         context = ""

#         try:
#             for i in range(request.iterations):
#                 logger.info(f"Starting iteration {i + 1}")
                
#                 result = await self.generate_content_iteration(
#                     prompt=request.prompt,
#                     genre=request.genre,
#                     context=context
#                 )
                
#                 results.append(result)
#                 context = result.story  # Use story as context for next iteration

#             # Clean up aiohttp session
#             if self._session and not self._session.closed:
#                 await self._session.close()

#             return results
            
#         except Exception as e:
#             logger.error(f"Pipeline execution failed: {str(e)}")
#             if self._session and not self._session.closed:
#                 await self._session.close()
#             raise

#     def validate_request(self, prompt: str, genre: str) -> bool:
#         """Validate user input"""
#         if not prompt.strip() or not genre.strip():
#             return False
            
#         # Check for minimum length
#         if len(prompt.strip()) < 3:
#             return False
            
#         return True








# import logging
# import os
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# import requests

# # Initialize logger
# logger = logging.getLogger(__name__)

# # Initialize OpenAI LLM
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
# llm = ChatOpenAI(
#     model_name="gpt-4o-mini",  
#     temperature=0.7,
#     openai_api_key=OPENAI_API_KEY,
#     max_retries=3,
#     verbose=True
# )

# # Define Prompt Templates
# base_prompt = PromptTemplate.from_template(
#     template="Create a {genre} story based on: {prompt}.\nThe story should be brief and descriptive and not more than 2-3 sentences long."
# )

# enhanced_prompt_template = PromptTemplate.from_template(
#     template="Enhance the following story for more vivid imagery and detail:\n{context}\nEnhanced story:"
# )

# # Function to generate content in multiple iterations
# def generate_content_pipeline(prompt, genre, colab_url, iterations=4):
#     """
#     Generate content (stories, images) for multiple iterations using LangChain.
#     """
#     results = []
#     context = ""
    
#     for i in range(iterations):
#         logger.info(f"--- Iteration {i + 1} ---")
        
#         # Step 1: Generate Story
#         story_prompt = base_prompt.format(genre=genre, prompt=prompt if i == 0 else context)
#         story_response = llm.invoke({"input": story_prompt})
#         story = story_response.content.strip()
#         logger.info(f"Iteration {i + 1} Story: {story}")
        
#         # Step 2: Generate Enhanced Context for Image
#         image_prompt = enhanced_prompt_template.format(context=story)
#         enhanced_story = llm.invoke({"input": image_prompt}).content.strip()
#         logger.info(f"Iteration {i + 1} Enhanced Context: {enhanced_story}")
        
#         # Step 3: Call Colab/AWS for Image Generation
#         try:
#             image_response = requests.post(f"{colab_url}/generate-image", json={"prompt": enhanced_story})
#             image_response.raise_for_status()
#             image_data = image_response.json().get("image_data")
#             logger.info(f"Iteration {i + 1} Image generated.")
#         except Exception as e:
#             logger.error(f"Error generating image at iteration {i + 1}: {str(e)}")
#             image_data = None
        
#         # Append iteration result
#         results.append({"story": story, "image_data": image_data, "enhanced_story": enhanced_story})
#         context = story  # Use story as context for the next iteration

#     return results










# import os
# import requests
# import logging
# # from langchain.llms import OpenAI
# from langchain_openai import OpenAI
# from langchain import LLMChain, PromptTemplate

# logger = logging.getLogger(__name__)

# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# MODEL_NAME = "gpt-4o-mini" 

# # Initialize the LLM
# llm = OpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model_name=MODEL_NAME,
#     temperature=0.7,
#     max_tokens=300
# )

# # Prompt templates
# enhance_prompt_template = PromptTemplate(
#     input_variables=["prompt", "genre"],
#     template="""
#     Enhance the following prompt for a {genre} story and image generation.
#     Include specific details about the setting, characters, and key elements.
#     The enhanced prompt should be suitable for both story writing and image creation.

#     Original prompt: {prompt}

#     Enhanced prompt:
#     """
# )
# enhance_chain = LLMChain(llm=llm, prompt=enhance_prompt_template)

# story_prompt_template = PromptTemplate(
#     input_variables=["genre", "enhanced_prompt", "previous_story_context", "previous_image_context"],
#     template="""
#     You are writing a {genre} story in multiple iterations.
#     So far, the previous part of the story is:
#     {previous_story_context}

#     The previous image context or description (if any) is:
#     {previous_image_context}

#     Continue the story in 2-3 sentences based on the following enhanced prompt:
#     {enhanced_prompt}

#     The new story snippet should logically continue from the previous story and maintain a consistent atmosphere and style.
#     """
# )
# story_chain = LLMChain(llm=llm, prompt=story_prompt_template)


# def generate_image(enhanced_prompt: str, colab_url: str) -> str:
#     # Calls the Colab endpoint to generate image
#     response = requests.post(f"{colab_url}/generate-image", json={"prompt": enhanced_prompt})
#     response.raise_for_status()
#     data = response.json()
#     img_data = data.get("image_data")
#     if not img_data:
#         raise ValueError("No image data received from Colab.")
#     return img_data

# def generate_voice(text: str, elevenlabs_api_key: str, voice_id="EXAVITQu4vr4xnSDxMaL"):
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
#     headers = {
#         "Accept": "audio/mpeg",
#         "Content-Type": "application/json",
#         "xi-api-key": elevenlabs_api_key
#     }
#     payload = {
#         "text": text,
#         "model_id": "eleven_monolingual_v1",
#         "voice_settings": {
#             "stability": 0.5,
#             "similarity_boost": 0.5
#         }
#     }

#     response = requests.post(url, json=payload, headers=headers)
#     response.raise_for_status()
#     return response.content  # raw audio data

# def run_iterations(prompt: str, genre: str, colab_url: str, elevenlabs_api_key: str, iterations: int = 4):
#     # 1. Enhance the prompt once at the start
#     logger.info("Starting prompt enhancement...")
#     enhanced_prompt = enhance_chain.run(prompt=prompt, genre=genre).strip()
#     logger.info(f"Enhanced Prompt: {enhanced_prompt}")

#     previous_story_context = ""
#     previous_image_context = ""

#     results = []

#     for i in range(1, iterations + 1):
#         logger.info(f"Starting iteration {i} of {iterations}")

#         # Generate Image
#         logger.info("Generating image...")
#         image_data = generate_image(enhanced_prompt, colab_url)
#         logger.info("Image generated successfully")

#         # Generate Story
#         logger.info("Generating story...")
#         story_part = story_chain.run(
#             genre=genre, 
#             enhanced_prompt=enhanced_prompt,
#             previous_story_context=previous_story_context,
#             previous_image_context=previous_image_context
#         ).strip()
#         logger.info(f"Story generated for iteration {i}: {story_part}")

#         # Generate Voice
#         logger.info("Generating voice...")
#         voice_data = generate_voice(story_part, elevenlabs_api_key)
#         logger.info("Voice generated successfully")

#         # Update context for next iteration
#         previous_story_context += "\n" + story_part
#         # Optional: You could derive a textual image description from the prompt or story 
#         # and store it as previous_image_context for the next iteration.
#         # For now, let's say the image context is just the last enhanced prompt.
#         previous_image_context = f"Last image was generated from prompt: {enhanced_prompt}"

#         # Store results of this iteration
#         results.append({
#             "iteration": i,
#             "image_data": image_data,
#             "story_part": story_part,
#             "voice_data": voice_data
#         })

#         # If you want to slightly modify the enhanced_prompt each iteration, you could 
#         # re-run enhancement with the updated context. For now, we keep it the same.

#     return {
#         "enhanced_prompt": enhanced_prompt,
#         "iterations_data": results
#     }

# def generate_content_flow(prompt: str, genre: str, colab_url: str, elevenlabs_api_key: str, iterations: int = 4):
#     return run_iterations(prompt, genre, colab_url, elevenlabs_api_key, iterations=iterations)
