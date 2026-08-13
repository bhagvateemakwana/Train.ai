from django.urls import path
from . import views

urlpatterns = [
    path('', views.RidgeRegressionView.as_view(), name="ridge_regression")
]
