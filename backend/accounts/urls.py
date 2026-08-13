from django.urls import path
from .views import register_view, login_view, logout_view, update_premium_status, update_liked_model, UserProfileView

urlpatterns = [
    path('signup/', register_view),
    path('signin/', login_view),
    path('signout/', logout_view),
    path('profile/', UserProfileView.as_view()),
    path('set-premium/', update_premium_status),
    path('update-liked-model/<str:model_id>/', update_liked_model, name='update_liked_model'),
]
