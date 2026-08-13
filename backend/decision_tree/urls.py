from django.urls import path
from . import views


urlpatterns = [
    path('', views.DecisionTreeView.as_view(), name='decision_tree')
]