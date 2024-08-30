# core/views.py
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect

from UserText.views import create_post


def home(request):
    # Call the create_post function to handle the form logic
    response = create_post(request)

    if isinstance(response, HttpResponseRedirect):  # If create_post returns a redirect response
        return response  # This means form submission was successful

    form = response  # If it's not a redirect, it's a form instance
    return render(request, 'core/home.html', {'form': form})  # Render the home template with the form



def terms_of_usage(request):
    return render(request, 'core/terms_of_usage.html')

def privacy_policy(request):
    return render(request, 'core/privacy_policy.html')