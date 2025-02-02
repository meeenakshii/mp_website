from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('graphical_method/', views.gp_index, name='graphical_method'),
    path('simplex_method/',views.simplex_method, name='simplex_method'),
    path('transportation_problem/', views.transportation_problem, name='transportation_problem'),
]
