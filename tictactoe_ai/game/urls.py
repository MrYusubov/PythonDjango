from django.urls import path
from .views import ai_move, health

urlpatterns = [
    path("health/", health),
    path("move/", ai_move),
]
