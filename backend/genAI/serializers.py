from rest_framework.renderers import JSONRenderer
import io
from rest_framework.parsers import JSONParser
from rest_framework import serializers
from .models import notes

class notesSerializers(serializers.ModelSerializer):
    class Meta:
        model = notes
        fields = '__all__'