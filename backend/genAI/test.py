from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.react_multi_hop.parsing import parse_answer_with_prefixes
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langsmith import Client
from langsmith.run_helpers import traceable, trace
import asyncio
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

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

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM errors during processing."""
        self.failed_requests += 1

class StoryIterationChain:
    def __init__(self):
        # Initialize callbacks
        self.token_callback = TokenUsageCallback()
        
        # Initialize LangSmith client
        self.client = Client()
        
        # Initialize LLM
        self.llm = ChatCohere(
            cohere_api_key=os.getenv("CO_API_KEY"),
            temperature=0.5,
            max_tokens=150,
            callbacks=[self.token_callback]
        )
        
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
            print(f"Error in generation: {str(e)}")
            return {
                "story": "Error occurred in story generation.",
                "image": "Error occurred in image description."
            }

    @traceable(run_type="chain")
    async def generate_full_story(self, initial_prompt: str, iterations: int = 4) -> list[Dict[str, str]]:
        """Generate a complete story with multiple iterations."""
        with trace(
            name="Full Story Generation",
            run_type="chain",
            project_name=os.getenv("LANGSMITH_PROJECT")
        ) as run:
            results = []
            previous_content = None
            
            for i in range(iterations):
                iteration_result = await self.generate_iteration(
                    initial_prompt if i == 0 else "", 
                    previous_content
                )
                results.append(iteration_result)
                previous_content = iteration_result
                
                # Add iteration metadata
                run.add_metadata({
                    f"iteration_{i+1}": {
                        "story": iteration_result["story"],
                        "image": iteration_result["image"]
                    }
                })
            
            return results

async def main():
    # Create and run the story chain
    story_chain = StoryIterationChain()
    
    initial_prompt = "a man inside a jungle"
    story_iterations = await story_chain.generate_full_story(initial_prompt)
    
    # Print results
    for i, iteration in enumerate(story_iterations, 1):
        print(f"\nIteration {i}:")
        print(f"Story: {iteration['story']}")
        print(f"Image: {iteration['image']}")

if __name__ == "__main__":
    asyncio.run(main())









# from langchain_cohere import ChatCohere
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_cohere.react_multi_hop.parsing import parse_answer_with_prefixes
# import asyncio
# from typing import Optional, Dict

# class StoryIterationChain:
#     def __init__(self, cohere_api_key):
#         self.llm = ChatCohere(
#             cohere_api_key=cohere_api_key,
#             temperature=0.5,
#             max_tokens=150  
#         )
        
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

#     async def generate_iteration(self, input_text: str, previous_content: Optional[Dict[str, str]] = None) -> Dict[str, str]:
#         try:
#             if previous_content is None:
#                 response = await self.llm.ainvoke(
#                     self.base_prompt.format_prompt(input_prompt=input_text).to_messages()
#                 )
#             else:
#                 response = await self.llm.ainvoke(
#                     self.continuation_prompt.format_prompt(
#                         previous_story=previous_content["story"]
#                     ).to_messages()
#                 )
            
#             # print(f"Raw response:\n{response.content}\n")
            
#             parsed_content = parse_answer_with_prefixes(response.content, self.prefixes)
            
#             # print(f"Parsed content:\n{parsed_content}\n")
            
#             return parsed_content
            
#         except Exception as e:
#             print(f"Error in generation: {str(e)}")
#             return {
#                 "story": "Error occurred in story generation.",
#                 "image": "Error occurred in image description."
#             }

#     async def generate_full_story(self, initial_prompt: str, iterations: int = 4) -> list[Dict[str, str]]:
#         results = []
#         previous_content = None
        
#         for i in range(iterations):
#             iteration_result = await self.generate_iteration(
#                 initial_prompt if i == 0 else "", 
#                 previous_content
#             )
#             results.append(iteration_result)
#             previous_content = iteration_result
            
#         return results

# async def main():
#     story_chain = StoryIterationChain(cohere_api_key="SDVrZC6I4V4yEVnzdc53Fo7yT2iJDhmjmbsUSEhh")
    
#     initial_prompt = "boat on a lake"
#     story_iterations = await story_chain.generate_full_story(initial_prompt)
    
#     for i, iteration in enumerate(story_iterations, 1):
#         print(f"\nIteration {i}:")
#         print(f"Story: {iteration['story']}")
#         print(f"Image: {iteration['image']}")

# if __name__ == "__main__":
#     asyncio.run(main())