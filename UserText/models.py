from django.db import models
from django.utils import timezone
from datetime import timedelta

class UserText(models.Model):
    text = models.TextField()
    expiry_date = models.DateTimeField(default=timezone.now() + timedelta(days=3)) #set expiry date 3 days after  posting
    ip_address = models.GenericIPAddressField()

    def __str__(self):
        return f"{self.text[:20]}... text from {self.ip_address} expires on {self.expiry_date}"