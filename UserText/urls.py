from django.urls import path
from . import views

urlpatterns = [
    path('post/<int:post_id>/', views.view_post, name='view_post'),
    path('posts/', views.all_posts, name='all_posts'),
]