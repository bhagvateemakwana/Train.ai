from django.urls import path
from . import views

urlpatterns = [
    path('', views.RandomForestView.as_view(), name='random_forest')
]
