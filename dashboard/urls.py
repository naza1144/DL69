from django.urls import path
from .views import *

urlpatterns = [
    path('', home),
    path('api/train', train),
    path('mlp', home_mlp),
    path('api/train-mlp', train_mlp)
]
