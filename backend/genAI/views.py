from django.contrib.auth.models import User
from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
import json
import traceback
from rest_framework import status
from rest_framework_simplejwt.views import TokenObtainPairView
from .models import notes
from django.http import JsonResponse
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
# from .services.langchain_service import generate_content_pipeline
from .services.langchain_service import LangChainService, ContentRequest
from .serializers import notesSerializers
import requests
from asgiref.sync import async_to_sync
from django.http import JsonResponse
import base64
import io
import cohere
import logging
from dotenv import load_dotenv 
import os

logger = logging.getLogger(__name__)
load_dotenv()

# ------------------------ APIKEYS ------------------------ #
COHERE_API_KEY = "D6fYNPT9Se1DEvbBk9umV6BTFKELycf16Te4RIlr"
ELEVENLABS_API_KEY = "sk_dff352e08e5cf56dea6532a60b600197d775dfd03e8b86db"
co = cohere.Client(COHERE_API_KEY)
# API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
# API_URL = "https://api-inference.huggingface.co/models/Lykon/DreamShaper"
# headers = {"Authorization": "Bearer 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"}
COLAB_URL = "https://87c7-35-185-226-172.ngrok-free.app"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not set")


# ------------------------ FUNCTIONS ------------------------ #
langchain_service = None
# use with colab or sagemaker
@csrf_exempt
def update_ngrok_url(request):
    global COLAB_URL  
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            ngrok_url = data.get('ngrok_url')
            if not ngrok_url:
                return JsonResponse({"error": "Ngrok URL is required"}, status=400)

            COLAB_URL = ngrok_url
            print(f"Received and updated Ngrok URL: {COLAB_URL}")

            return JsonResponse({"message": "Ngrok URL updated successfully"}, status=200)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)

def get_langchain_service():
    """Get or create LangChain service with current COLAB_URL"""
    global langchain_service, COLAB_URL, OPENAI_API_KEY
    
    logger.info(f"Initializing LangChain service with COLAB_URL: {COLAB_URL}")
    logger.info(f"OPENAI_API_KEY set: {bool(OPENAI_API_KEY)}")
    
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable must be set")
        
    if langchain_service is None or langchain_service.colab_url != COLAB_URL:
        try:
            logger.info("Creating new LangChain service instance")
            langchain_service = LangChainService(
                openai_api_key=OPENAI_API_KEY,
                colab_url=COLAB_URL
            )
            logger.info("LangChain service created successfully")
        except Exception as e:
            logger.error(f"Error creating LangChain service: {str(e)}")
            raise

    return langchain_service


class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        # Add custom claims
        token['username'] = user.username
        token['password'] = user.password
        # ...
        return token

class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

@api_view(['GET'])
def getRoutes(request):
    routes = [
        '/api/token',
        '/api/token/refresh',
    ]
    return Response(routes)


@csrf_exempt
def generate_content(request):
    if request.method == 'POST':
        try:
            # Print the raw request data for debugging
            logger.info(f"Raw request data: {request.body}")
            
            # Check if OpenAI API key is set
            if not OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY not set")
                return JsonResponse({
                    "error": "OPENAI_API_KEY environment variable must be set"
                }, status=500)
            
            # Check if COLAB_URL is set
            if not COLAB_URL:
                logger.error("COLAB_URL not set")
                return JsonResponse({
                    "error": "Colab URL not set. Update via /update-ngrok-url/."
                }, status=500)

            # Parse request data
            try:
                data = json.loads(request.body)
                logger.info(f"Parsed request data: {data}")

                content_request = ContentRequest(
                    prompt=data.get("prompt"),
                    genre=data.get("genre"),
                    iterations=data.get("iterations", 4)
                )
                logger.info(f"Created content request: {content_request}")

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                return JsonResponse({"error": "Invalid JSON format."}, status=400)
            except ValueError as e:
                logger.error(f"Content request creation error: {str(e)}")
                return JsonResponse({"error": f"Invalid request format: {str(e)}"}, status=400)

            # Get LangChain service
            try:
                logger.info("Getting LangChain service")
                service = get_langchain_service()
                logger.info("LangChain service initialized")
            except ValueError as e:
                logger.error(f"LangChain service initialization error: {str(e)}")
                return JsonResponse({"error": str(e)}, status=500)
            
            # Generate content
            try:
                logger.info("Starting content generation")
                
                # Create new event loop for async operations
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    results = loop.run_until_complete(
                        service.generate_content_pipeline(content_request)
                    )
                finally:
                    # Cleanup
                    loop.run_until_complete(service.cleanup())
                    loop.close()
                
                logger.info("Content generation completed")

                # Convert results to JSON-serializable format
                serialized_results = [
                    {
                        "story": r.story,
                        "enhanced_story": r.enhanced_story,
                        "image_url": r.image_url,
                        "iteration": r.iteration
                    }
                    for r in results
                ]

                return JsonResponse({
                    "results": serialized_results
                }, status=200)

            except Exception as e:
                logger.error(f"Content generation error: {str(e)}")
                return JsonResponse({"error": str(e)}, status=500)

        except Exception as e:
            logger.error(f"Unexpected error in generate_content: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                "error": f"Unexpected error: {str(e)}"
            }, status=500)

    return JsonResponse({
        "error": "Invalid request method."
    }, status=405)
# @csrf_exempt
# def generate_content(request):
#     if request.method == 'POST':
#         try:
#             # Print the raw request data for debugging
#             logger.info(f"Raw request data: {request.body}")
            
#             # Check if OpenAI API key is set
#             if not OPENAI_API_KEY:
#                 logger.error("OPENAI_API_KEY not set")
#                 return JsonResponse({
#                     "error": "OPENAI_API_KEY environment variable must be set"
#                 }, status=500)
            
#             # Check if COLAB_URL is set
#             if not COLAB_URL:
#                 logger.error("COLAB_URL not set")
#                 return JsonResponse({
#                     "error": "Colab URL not set. Update via /update-ngrok-url/."
#                 }, status=500)

#             # Parse request data
#             try:
#                 data = json.loads(request.body)
#                 logger.info(f"Parsed request data: {data}")

#                 # Basic input validation
#                 if not data.get("prompt") or not data.get("prompt").strip():
#                     return JsonResponse({
#                         "error": "Prompt cannot be empty"
#                     }, status=400)
                
#                 if not data.get("genre") or not data.get("genre").strip():
#                     return JsonResponse({
#                         "error": "Genre cannot be empty"
#                     }, status=400)

#                 # Create content request
#                 content_request = ContentRequest(
#                     prompt=data.get("prompt"),
#                     genre=data.get("genre"),
#                     iterations=data.get("iterations", 4)
#                 )
#                 logger.info(f"Created content request: {content_request}")

#             except json.JSONDecodeError as e:
#                 logger.error(f"JSON decode error: {str(e)}")
#                 return JsonResponse({"error": "Invalid JSON format."}, status=400)
#             except ValueError as e:
#                 logger.error(f"Content request creation error: {str(e)}")
#                 return JsonResponse({"error": f"Invalid request format: {str(e)}"}, status=400)

#             # Get LangChain service
#             try:
#                 logger.info("Getting LangChain service")
#                 service = get_langchain_service()
#                 logger.info("LangChain service initialized")
#             except ValueError as e:
#                 logger.error(f"LangChain service initialization error: {str(e)}")
#                 return JsonResponse({"error": str(e)}, status=500)
            
#             # Generate content
#             try:
#                 logger.info("Starting content generation")
#                 results = async_to_sync(service.generate_content_pipeline)(content_request)
#                 logger.info("Content generation completed")

#                 # Convert results to JSON-serializable format
#                 serialized_results = [
#                     {
#                         "story": r.story,
#                         "enhanced_story": r.enhanced_story,
#                         "image_url": r.image_url,
#                         "iteration": r.iteration
#                     }
#                     for r in results
#                 ]

#                 return JsonResponse({
#                     "results": serialized_results
#                 }, status=200)

#             except Exception as e:
#                 logger.error(f"Content generation error: {str(e)}")
#                 return JsonResponse({"error": str(e)}, status=500)

#         except Exception as e:
#             logger.error(f"Unexpected error in generate_content: {str(e)}")
#             import traceback
#             traceback.print_exc()  # Print full traceback
#             return JsonResponse({
#                 "error": f"Unexpected error: {str(e)}"
#             }, status=500)

#     return JsonResponse({
#         "error": "Invalid request method."
#     }, status=405)
    

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def getUserDetails(request):
    user = request.user
    user_data = {
        'id': user.id,
        'username': user.username,
        # Add other fields if needed
    }
    return Response(user_data)

@api_view(['POST'])
def create(request):
    data = request.data
    username = data.get("username", "").lower()
    password = data.get("password", "")
    if User.objects.filter(username=username).exists():
        return Response({"error": "USER ALREADY EXISTS"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        user = User.objects.create_user(username=username,password=password)
        user.save()
        return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def getNotes(request):
    user = request.user
    Notes = user.notes_set.all()  # Use the related_name defined in the Note model
    serializer = notesSerializers(Notes, many=True)
    return Response(serializer.data)






 # ------------------------------------  EXTRA STUFF ------------------------------------ #
 
 
# @csrf_exempt
# def generate_content(request):
#     if request.method == 'POST':
#         try:
#             # Print the raw request data for debugging
#             print("Raw request data:", request.body)
            
#             # Check if OpenAI API key is set
#             if not OPENAI_API_KEY:
#                 logger.error("OPENAI_API_KEY not set")
#                 return JsonResponse({
#                     "error": "OPENAI_API_KEY environment variable must be set"
#                 }, status=500)
            
#             # Check if COLAB_URL is set
#             if not COLAB_URL:
#                 logger.error("COLAB_URL not set")
#                 return JsonResponse({
#                     "error": "Colab URL not set. Update via /update-ngrok-url/."
#                 }, status=500)

#             # Parse request data
#             try:
#                 data = json.loads(request.body)
#                 logger.info(f"Parsed request data: {data}")
#             except json.JSONDecodeError as e:
#                 logger.error(f"JSON decode error: {str(e)}")
#                 return JsonResponse({"error": "Invalid JSON format."}, status=400)

#             # Create content request
#             try:
#                 content_request = ContentRequest(
#                     prompt=data.get("prompt"),
#                     genre=data.get("genre"),
#                     iterations=data.get("iterations", 4)
#                 )
#                 logger.info(f"Created content request: {content_request}")
#             except Exception as e:
#                 logger.error(f"Content request creation error: {str(e)}")
#                 return JsonResponse({"error": f"Invalid request format: {str(e)}"}, status=400)

#             # Get LangChain service
#             try:
#                 service = get_langchain_service()
#                 logger.info("LangChain service initialized")
#             except ValueError as e:
#                 logger.error(f"LangChain service initialization error: {str(e)}")
#                 return JsonResponse({"error": str(e)}, status=500)
            
#             # Validate input
#             if not service.validate_request(content_request.prompt, content_request.genre):
#                 logger.error("Input validation failed")
#                 return JsonResponse({
#                     "error": "Invalid input parameters"
#                 }, status=400)

#             # Generate content
#             try:
#                 logger.info("Starting content generation")
#                 results = async_to_sync(service.generate_content_pipeline)(content_request)
#                 logger.info("Content generation completed")
#             except Exception as e:
#                 logger.error(f"Content generation error: {str(e)}")
#                 traceback.print_exc()  # Print full traceback
#                 return JsonResponse({"error": str(e)}, status=500)

#             # Serialize results
#             try:
#                 serialized_results = [
#                     {
#                         "story": r.story,
#                         "enhanced_story": r.enhanced_story,
#                         "image_url": r.image_url
#                     }
#                     for r in results
#                 ]
#                 logger.info("Results serialized successfully")
#             except Exception as e:
#                 logger.error(f"Serialization error: {str(e)}")
#                 return JsonResponse({"error": "Error serializing results"}, status=500)

#             return JsonResponse({
#                 "results": serialized_results
#             }, status=200)

#         except Exception as e:
#             logger.error(f"Unexpected error in generate_content: {str(e)}")
#             traceback.print_exc()  # Print full traceback
#             return JsonResponse({
#                 "error": f"Unexpected error: {str(e)}"
#             }, status=500)

#     return JsonResponse({
#         "error": "Invalid request method."
#     }, status=405)
    
# @csrf_exempt
# def enhance_prompt(prompt, genre):
#     try:
#         enhancement_prompt = f"""
#         Enhance the following prompt for a {genre} story and image generation. 
#         Include specific details about the setting, characters, and key elements. 
#         The enhanced prompt should be suitable for both story writing and image creation.
        
#         Original prompt: {prompt}
        
#         Enhanced prompt:
#         """
#         response = co.generate(
#             model='command',
#             prompt=enhancement_prompt,
#             max_tokens=150, 
#             temperature=0.7,  # 0.0 is least creative, 1.0 is most creative
#             k=0,  
#             stop_sequences=[],
#             return_likelihoods='NONE')
#         logger.info(f"Enhanced prompt response: {response.generations[0].text.strip()}")
#         return response.generations[0].text.strip()
#     except Exception as e:
#         logger.error(f"Error enhancing prompt: {str(e)}")
#         return prompt  
    
# def generate_story(enhanced_prompt, genre):
#     try:
#         story_prompt = f"""
#         Write a very short {genre} story based on this prompt: {enhanced_prompt}
#         The story should be only 2-3 sentences long.
#         Focus on describing the key elements and atmosphere, avoiding any complex plot.
#         """
#         response = co.generate(
#             model='command',
#             prompt=story_prompt,
#             max_tokens=50,  
#             temperature=0.6,
#             k=0,
#             stop_sequences=[],
#             return_likelihoods='NONE')
#         return response.generations[0].text.strip()
#     except Exception as e:
#         print(f"Error generating story: {str(e)}")
#         return None


# @csrf_exempt
# def generate_content(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get("prompt")
#             genre = data.get("genre")
            
#             if not prompt or not genre:
#                 return JsonResponse({"error": "Prompt and genre are required."}, status=400)
            
#             if not COLAB_URL:
#                 return JsonResponse({"error": "Colab URL not set. Update via /update-ngrok-url/."}, status=500)
            
#             # Call the content generation pipeline
#             results = generate_content_pipeline(prompt, genre, COLAB_URL)
#             return JsonResponse({"results": results}, status=200)
        
#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON format."}, status=400)
#         except Exception as e:
#             logger.error(f"Error in content generation: {str(e)}")
#             return JsonResponse({"error": str(e)}, status=500)
#     return JsonResponse({"error": "Invalid request method."}, status=405)

# @csrf_exempt
# def generate_content(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')
#             genre = data.get('genre')
#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON"}, status=400)

#         if not prompt or not genre:
#             return JsonResponse({"error": "Prompt and genre are required"}, status=400)

#         try:
#             # Enhance the prompt
#             enhanced_prompt = enhance_prompt(prompt, genre)
            
#             # Generate image using the enhanced prompt
#             image_response = requests.post(f"{COLAB_URL}/generate-image", json={"prompt": enhanced_prompt})
#             image_response.raise_for_status()
#             image_data = image_response.json().get('image_data')

#             # Generate story using the enhanced prompt
#             story = generate_story(enhanced_prompt, genre)

#             if image_data and story:
#                 return JsonResponse({
#                     "image_data": image_data, 
#                     "story": story,
#                     "enhanced_prompt": enhanced_prompt  # Optionally include this
#                 }, status=200)
#             else:
#                 return JsonResponse({"error": "Failed to generate content"}, status=500)
#         except requests.exceptions.RequestException as e:
#             return JsonResponse({"error": f"Error from Colab: {str(e)}"}, status=500)

#     return JsonResponse({"error": "Invalid request method"}, status=405)

# @csrf_exempt
# def generate_voice(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             text = data.get('text')
#             voice_id = data.get('voice_id', 'EXAVITQu4vr4xnSDxMaL')  # Default voice ID
            
#             if not text:
#                 return JsonResponse({"error": "Text is required"}, status=400)

#             url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
            
#             headers = {
#                 "Accept": "audio/mpeg",
#                 "Content-Type": "application/json",
#                 "xi-api-key": ELEVENLABS_API_KEY
#             }

#             data = {
#                 "text": text,
#                 "model_id": "eleven_monolingual_v1",
#                 "voice_settings": {
#                     "stability": 0.5,
#                     "similarity_boost": 0.5
#                 }
#             }

#             response = requests.post(url, json=data, headers=headers)
            
#             if response.status_code == 200:
#                 return JsonResponse({
#                     "audio_data": response.content.decode('ISO-8859-1'),
#                     "content_type": response.headers.get('Content-Type')
#                 })
#             else:
#                 return JsonResponse({"error": "Failed to generate voice"}, status=500)

#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON"}, status=400)
#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)

#     return JsonResponse({"error": "Invalid request method"}, status=405)
 
#  @csrf_exempt
# def generate_image(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')
#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON"}, status=400)

#         if not prompt:
#             return JsonResponse({"error": "Prompt is required"}, status=400)

#         try:
#             response = requests.post(f"{COLAB_URL}/generate-image", json={"prompt": prompt})
#             response.raise_for_status()
#             return JsonResponse(response.json(), status=200)
#         except requests.exceptions.RequestException as e:
#             return JsonResponse({"error": f"Error from Colab: {str(e)}"}, status=500)

#     return JsonResponse({"error": "Invalid request method"}, status=405)

# def test_post(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             return JsonResponse({"message": "Data received", "data": data}, status=200)
#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON"}, status=400)
#     return JsonResponse({"error": "Only POST requests are allowed"}, status=405)

# @csrf_exempt
# def generate_image(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')
#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON"}, status=400)

#         if not prompt:
#             return JsonResponse({"error": "Prompt is required"}, status=400)

#         try:
#             response = requests.post(f"{COLAB_URL}/generate-image", json={"prompt": prompt})
#             response.raise_for_status()
#             return JsonResponse(response.json(), status=200)
#         except requests.exceptions.RequestException as e:
#             return JsonResponse({"error": f"Error from Colab: {str(e)}"}, status=500)

#     return JsonResponse({"error": "Invalid request method"}, status=405)

# USE THIS WITH HUGGINGFACE INFERENCE API FOR TESTING FRONTEND AND BACKEND
# @csrf_exempt
# def generate_image(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')
#             logger.info(f"Received prompt: {prompt}")
#         except json.JSONDecodeError as e:
#             logger.error(f"JSON decode error: {str(e)}")
#             return JsonResponse({"error": "Invalid JSON"}, status=400)

#         if not prompt:
#             logger.error("No prompt provided")
#             return JsonResponse({"error": "Prompt is required"}, status=400)

#         try:
#             logger.info(f"Sending request to Hugging Face API with prompt: {prompt}")
#             response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
#             response.raise_for_status()

#             image_bytes = response.content
#             image_base64 = base64.b64encode(image_bytes).decode('utf-8')
#             logger.info("Successfully generated image")
#             return JsonResponse({"image_data": image_base64}, status=200)
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error from Hugging Face API: {str(e)}")
#             return JsonResponse({"error": f"Error from Hugging Face API: {str(e)}"}, status=500)
#         except Exception as e:
#             logger.error(f"Unexpected error: {str(e)}")
#             return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)

# def test_post(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             return JsonResponse({"message": "Data received", "data": data}, status=200)
#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON"}, status=400)
#     return JsonResponse({"error": "Only POST requests are allowed"}, status=405)



#     logger.error("Invalid request method")
#     return JsonResponse({"error": "Invalid request method"}, status=405)

#     return JsonResponse({"error": "Invalid request method"}, status=405)
#     if request.method == "POST":
#         try:
#             # Parse JSON data from the request body
#             data = json.loads(request.body)
#             prompt = data.get('prompt')

#             if not prompt:
#                 return JsonResponse({"error": "Prompt is required"}, status=400)

#             # Send the prompt to the Hugging Face API
#             response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

#             if response.status_code == 200:
#                 image_data = response.content
#                 image_base64 = base64.b64encode(image_data).decode('utf-8')
#                 return JsonResponse({"image_data": image_base64}, status=200)
#             else:
#                 return JsonResponse({"error": "Failed to generate image"}, status=500)
#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)

#     return JsonResponse({"error": "Invalid request method"}, status=405)

# def generate_image(request):
#     prompt = request.data.get('prompt')
    
#     if not prompt:
#         return JsonResponse({"error": "Prompt is required"}, status=400)

#     try:
#         response = requests.post("https://7db6-3-20-229-229.ngrok-free.app/", json={"prompt": prompt})

#         if response.status_code == 200:
#             image_data = response.json().get("image_data")
#             return JsonResponse({"image_data": image_data}, status=200)
#         else:
#             return JsonResponse({"error": "Failed to generate image"}, status=500)
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)

# @api_view(['GET'])
# def getUsers(request):
#     users = User.objects.all()
#     serializer = UserSerializer(users, many=True)
#     return Response(serializer.data)


# from django.shortcuts import render
# from rest_framework import status
# from rest_framework.decorators import api_view, permission_classes
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.response import Response
# from .serializers import userSerializers
# from .models import USERS
# from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
# from rest_framework_simplejwt.views import TokenObtainPairView

# class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
#     @classmethod
#     def get_token(cls, USERS):
#         token = super().get_token(USERS)
#         # Add custom claims
#         token['username'] = USERS.username
#         token['password'] = USERS.password
#         # ...
#         return token

# class MyTokenObtainPairView(TokenObtainPairView):
#     serializer_class = MyTokenObtainPairSerializer

# @api_view(['GET'])
# def getRoutes(request):
#     routes = [
#         '/api/token',
#         '/api/token/refresh',
#     ]
#     return Response(routes)

# @api_view(['GET'])
# def getUSERS(request):
#     users = USERS.objects.all()
#     serializer = userSerializers(users, many=True)
#     return Response(serializer.data)




# @api_view(['GET'])
# def get_routes(request):
#     routes = {
#         "list":"/list",
#         "create":"/create",
#         "update":"/update",
#         "delete":"/delete"
#     }
#     return Response(routes)

# class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
#     @classmethod
#     def get_token(cls, user):
#         token = super().get_token(user)

#         # Add custom claims
#         token['name'] = user.name
#         # ...

#         return token

# class MyTokenObtainPairView(TokenObtainPairView):
#     serializer_class = MyTokenObtainPairSerializer
    
# @api_view(['GET'])
# def getUSERS(request):
#     if request.method == "GET":
#         users = USERS.objects.all()
#         serializer = userSerializers(users, many=True)
#         return Response(serializer.data)