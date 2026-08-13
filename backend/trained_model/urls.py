from django.urls import path
from . import views

urlpatterns = [
    path('detail/<str:pk>/', views.ModelDetailView.as_view(), name='model_detail'),
    path('', views.ModelListView.as_view(), name='model_list'),
    path('user/', views.UserTrainedModelView.as_view(), name='user_trained_models'),
    path('update-model/<str:pk>/', views.ModelUpdateView.as_view(), name='update_model'),
    path('user/liked-models/', views.getUserLikedModels, name='user_liked_models'),
    path('report/<uuid:model_id>/', views.download_model_report, name='download_model_report'),
    path('train/', views.TrainModelView.as_view(), name='train_model'),
]
