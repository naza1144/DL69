from django.urls import path
from .views import home, train
from .views import home_mlp, train_mlp

urlpatterns = [
    path('', home),
    path('api/train', train),

    path('mlp', home_mlp),
    path('api/train_mlp', train_mlp)
]
