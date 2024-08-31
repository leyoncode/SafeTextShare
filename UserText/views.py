from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from django.contrib import messages
from django.conf import settings

from .forms import UserTextForm
from .models import UserText

import os
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the model and vectorizer once when the module is imported
model_path = os.path.join(settings.BASE_DIR, 'static/models/saved_model_knn.pkl')
vectorizer_path = os.path.join(settings.BASE_DIR, 'static/models/vectorizer.pkl')

# Normalize the paths
model_path = os.path.normpath(model_path)
vectorizer_path = os.path.normpath(vectorizer_path)


with open(model_path, 'rb') as model_file:
    text_model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(words)
    return cleaned_text


def create_post(request):
    if request.method == 'POST':
        form = UserTextForm(request.POST)
        if form.is_valid():
            post_text = form.cleaned_data['text']

            # Preprocess the text
            cleaned_text = preprocess_text(post_text)

            # Vectorize the cleaned text
            transformed_text = vectorizer.transform([cleaned_text])

            # Use the model to predict
            prediction = text_model.predict(transformed_text)[0]  # Assuming model returns 1 for flagging

            if prediction == 1:  # If model flags the text
                messages.error(request, 'Our AI detects your post as either violating the terms of use,\
                                        is malicious, spam, or unwanted text and cannot be saved.')
                return redirect('home')

            # If no violation, proceed to save
            post = form.save(commit=False)
            post.expiry_date = timezone.now() + timezone.timedelta(days=3)
            post.ip_address = request.META.get('REMOTE_ADDR')
            post.save()

            messages.success(request, 'Your post has been successfully created!')
            return redirect('view_post', post_id=post.id)
        # else:
        #     # Form is not valid, capture error messages
        #     for field, errors in form.errors.items():
        #         for error in errors:
        #             messages.error(request, f"{field}: {error}")

    else:
        form = UserTextForm()

    return form


def view_post(request, post_id):
    post = get_object_or_404(UserText, id=post_id)

    # Check if the post has expired
    if post.expiry_date < timezone.now():
        # If expired, show a message and redirect to another page
        messages.error(request, 'The post you tried to view is no longer available.')
        return redirect('all_posts')  # Redirect to the 'all_posts' page or any other page

    return render(request, 'UserText/view_post.html', {'post': post})


def all_posts(request):
    current_time = timezone.now()
    posts = UserText.objects.filter(expiry_date__gt=current_time)
    return render(request, 'UserText/all_posts.html', {'posts': posts})