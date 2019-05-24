from django.forms import ModelForm
from models import *
 class Meta:
  model = Doktor
                fields = ['adi','soyadi','yas','telefon','eposta']
 def clean_telefon(self):
  tel = self.cleaned_data['telefon']
  if tel != "":
   if len(tel) != 11:
    raise forms.ValidationError('Telefon numarasi 11 karakter olmalidir.')
  return tel