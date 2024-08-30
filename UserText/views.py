from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from .models import UserText

def view_post(request, post_id):
    post = get_object_or_404(UserText, id=post_id)
    return render(request, 'usertest/view_post.html', {'post': post})

def all_posts(request):
    current_time = timezone.now()
    posts = UserText.objects.filter(expiry_date__gt=current_time)
    return render(request, 'usertest/all_posts.html', {'posts': posts})