from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone

from .forms import UserTextForm
from .models import UserText

def view_post(request, post_id):
    post = get_object_or_404(UserText, id=post_id)
    return render(request, 'UserText/view_post.html', {'post': post})

def all_posts(request):
    current_time = timezone.now()
    posts = UserText.objects.filter(expiry_date__gt=current_time)
    return render(request, 'UserText/all_posts.html', {'posts': posts})


def create_post(request):
    if request.method == 'POST':
        form = UserTextForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.expiry_date = timezone.now() + timezone.timedelta(days=3)
            post.ip_address = request.META.get('REMOTE_ADDR')
            # print(post) #debug
            post.save()
            return redirect(f'/post/{post.id}/')  # Redirect after successful form submission
    else:
        form = UserTextForm()

    return form  # Return the form instance