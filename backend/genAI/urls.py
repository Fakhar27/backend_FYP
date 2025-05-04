from django.urls import path
from . import views
from .views import MyTokenObtainPairView
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('', views.getRoutes, name='routes'),
    path('notes/', views.getNotes, name='users'),
    path('token/', MyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('user/', views.getUserDetails, name='user_details'), 
    # path('test-post/', views.test_post, name='test_post'),
    path('register/',views.create,name="create"),
    path('generate-content/',views.generate_content,name="generate_content"),
    # path('generate-voice/',views.generate_voice,name="generate_voice"),
    # path('generate-image/', views.generate_image, name='generate_image'),
    path('update-ngrok-url/',views.update_ngrok_url,name='update_ngrok_url'),
    path('generate-voice/',views.generate_voice,name="generate_voice"),
    path('update-ngrok-url2/', views.update_ngrok_url_voice, name='update_ngrok_url'),
    path('update-ngrok-url3/', views.update_ngrok_url_whisper, name='update_ngrok_url_whisper'),
    path('wan_video_generation_request/',views.wan_video_generation_request, name='wan_video_generation_request'),
    path('wan_video_generation_request_complete_pipeline/',views.wan_video_generation_request_complete_pipeline, name='wan_video_generation_request'),
]