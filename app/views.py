from re import template
from django.shortcuts import redirect, render
from django.views.generic.base import (
    TemplateView
)
from django.http import HttpResponse
from .forms import PhotoForm
from .models import Photo
from django.template import loader
from .base64_img import pil_to_base64
from PIL import Image


def home(request):
    template = loader.get_template('home.html')
    context = {'form': PhotoForm()}
    return HttpResponse(template.render(context, request))

def instance(request):
    if not request.method == 'POST':
        return redirect('app:home')
    form = PhotoForm(request.POST, request.FILES)
    if not form.is_valid():
        raise ValueError('Formが不正です')
    photo = Photo(image=form.cleaned_data['image'])
    result_image = photo.predict()
    file = Image.fromarray(result_image)
    result_img = pil_to_base64(file)
    template = loader.get_template('instance.html')
    context = {
        'photo_name':photo.image.name,
        'photo_data':photo.img_src(),
        'result_img':result_img
    }
    return HttpResponse(template.render(context,request))


class About(TemplateView):
    template_name = 'about.html'


# エラーハンドリング
def page_not_found(request, exception):
    return redirect(request, '404.html', status=404)

def server_error(request):
    return render(request, '500.html', status=500)


