from django.urls import path
from .views import home, home_mlp, train, train_mlp

urlpatterns = [
    path('', home, name='home'),
    path('index.html', home, name='index'),
    path('mlp', home_mlp, name='home_mlp'),
    path('mlp/', home_mlp, name='home_mlp_slash'),
    path('train', train, name='train'),
    path('train/', train, name='train_slash'),
    path('api/train', train, name='api_train'),
    path('train-mlp', train_mlp, name='train_mlp'),
    path('train-mlp/', train_mlp, name='train_mlp_slash'),
    path('api/train-mlp', train_mlp, name='api_train_mlp'),
]
