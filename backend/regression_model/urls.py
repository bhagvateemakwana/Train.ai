from django.urls import path
from . import views

urlpatterns = [
    path('linear/', views.LinearRegressionView.as_view(), name='linear_regression'),
    path('polynomial/', views.PolynomialRegressionView.as_view(), name='polynomial_regression'),
]