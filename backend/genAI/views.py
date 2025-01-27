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
    Notes = user.notes_set.all()  
    serializer = notesSerializers(Notes, many=True)
    return Response(serializer.data)
