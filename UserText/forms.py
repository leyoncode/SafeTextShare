# UserText/forms.py

from django import forms
from .models import UserText

class UserTextForm(forms.ModelForm):
    class Meta:
        model = UserText
        fields = ['text']
        widgets = {
            'text': forms.Textarea(attrs={'placeholder': 'Enter your text here...'}),
        }