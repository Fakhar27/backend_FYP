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
