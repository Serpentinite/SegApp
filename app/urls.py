from re import template
from django.urls import path
from . import views


app_name = 'app'
urlpatterns = [
    path('', views.home, name='home'),
    path('instance/', views.instance, name='instance'),
    path('about/',views.About.as_view(), name='about')
]


