from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User

class notes(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    description = models.TextField()    