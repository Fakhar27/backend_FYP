from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User

class notes(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    description = models.TextField()    

# class USERS(models.Model):
#     name = models.CharField(max_length=30)
#     password = models.CharField(max_length=30)
#     confirmpassword = models.CharField(max_length=30)
    
#     def __str__(self):
#         return self.name
    
    

# class Teacher(models.Model):
#     name = models.CharField(max_length=80)
#     age = models.IntegerField()