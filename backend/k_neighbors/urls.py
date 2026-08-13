from django.urls import path
from . import views

urlpatterns = [
    path('', views.KNeighborsView.as_view(), name='k_neighbors')
]
