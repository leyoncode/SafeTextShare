from django.urls import path, include

from UserText.models import UserText
from core import views

urlpatterns = [
    path('', views.home, name='home'),
    path('terms-of-usage/', views.terms_of_usage, name='terms_of_usage'),
    path('privacy-policy/', views.privacy_policy, name='privacy_policy'),

    path('', include('UserText.urls'))
]