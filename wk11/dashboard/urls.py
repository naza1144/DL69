from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.index, name='index'),
    path('train/stream/', views.train_stream, name='train_stream'),
    path('api/model-info/', views.model_info, name='model_info'),
]
