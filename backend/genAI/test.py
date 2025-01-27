from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.react_multi_hop.parsing import parse_answer_with_prefixes
import asyncio
from typing import Optional, Dict

class StoryIterationChain:
    def __init__(self, cohere_api_key):
        self.llm = ChatCohere(
            cohere_api_key=cohere_api_key,
            temperature=0.5,
            max_tokens=150  
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

    async def generate_iteration(self, input_text: str, previous_content: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        try:
            if previous_content is None:
                response = await self.llm.ainvoke(
                    self.base_prompt.format_prompt(input_prompt=input_text).to_messages()
                )
            else:
                response = await self.llm.ainvoke(
                    self.continuation_prompt.format_prompt(
                        previous_story=previous_content["story"]
                    ).to_messages()
                )
            
            # print(f"Raw response:\n{response.content}\n")
            
            parsed_content = parse_answer_with_prefixes(response.content, self.prefixes)
            
            # print(f"Parsed content:\n{parsed_content}\n")
            
            return parsed_content
            
        except Exception as e:
            print(f"Error in generation: {str(e)}")
            return {
                "story": "Error occurred in story generation.",
                "image": "Error occurred in image description."
            }

    async def generate_full_story(self, initial_prompt: str, iterations: int = 4) -> list[Dict[str, str]]:
        results = []
        previous_content = None
        
        for i in range(iterations):
            iteration_result = await self.generate_iteration(
                initial_prompt if i == 0 else "", 
                previous_content
            )
            results.append(iteration_result)
            previous_content = iteration_result
            
        return results

async def main():
    story_chain = StoryIterationChain(cohere_api_key="SDVrZC6I4V4yEVnzdc53Fo7yT2iJDhmjmbsUSEhh")
    
    initial_prompt = "a man inside a jungle"
    story_iterations = await story_chain.generate_full_story(initial_prompt)
    
    for i, iteration in enumerate(story_iterations, 1):
        print(f"\nIteration {i}:")
        print(f"Story: {iteration['story']}")
        print(f"Image: {iteration['image']}")

if __name__ == "__main__":
    asyncio.run(main())