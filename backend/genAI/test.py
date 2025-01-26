from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

def test_langchain():
    try:
        # Test imports
        print("✓ Basic imports successful")
        
        # Test prompt template
        prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
        print("✓ PromptTemplate created successfully")
        
        # Test environment
        load_dotenv()  # Load environment variables from .env file
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("✓ OpenAI API key found in environment")
            
            # Test actual API call
            llm = ChatOpenAI()
            result = llm.invoke("Say hello!")
            print("✓ Successfully made API call to OpenAI")
            print(f"\nAPI Response: {result.content}")
        else:
            print("\n⚠ OpenAI API key not found. Please check:")
            print("1. Does .env file exist in the current directory?")
            print("2. Does it contain OPENAI_API_KEY=sk-...?")
            print("3. Current directory:", os.getcwd())
            print("4. Environment variables loaded:", dict(os.environ))
        
        return True
        
    except Exception as e:
        print(f"\n⚠ Error occurred: {str(e)}")
        if "api_key" in str(e).lower():
            print("\nAPI Key Error Tips:")
            print("1. Create a .env file in:", os.getcwd())
            print("2. Add this line: OPENAI_API_KEY=sk-your-key-here")
            print("3. Make sure the key starts with 'sk-'")
        return False

if __name__ == "__main__":
    test_langchain()