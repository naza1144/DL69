from django.urls import path
from .views import home, train, training, predict_mlp, mlp_list, mlp_detail, mlp_predict

urlpatterns = [
    path('', home),
    path('training/', training, name='training'),
    path('train/', training, name='train'),  # alias for landing page per request
    path('predict/mlp/', predict_mlp, name='predict_mlp'),
    path('api/train', train),
    path('api/mlp/models', mlp_list, name='mlp_list'),
    path('api/mlp/models/<int:pk>', mlp_detail, name='mlp_detail'),
    path('api/predict/mlp', mlp_predict, name='mlp_predict'),
]
