# UserText/forms.py

from django import forms
from django.core.validators import MinLengthValidator

from .models import UserText

class UserTextForm(forms.ModelForm):
    text = forms.CharField(
        widget=forms.Textarea(attrs={'placeholder': 'Enter your text here...'}),
        validators=[MinLengthValidator(10)]
    )

    class Meta:
        model = UserText
        fields = ['text']