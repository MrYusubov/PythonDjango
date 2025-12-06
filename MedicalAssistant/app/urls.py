from django.urls import path
from .views import form_view, result_view, admin_dashboard

urlpatterns = [
    path("", form_view, name="form"),
    path("result/", result_view, name="result"),
    path("dashboard/", admin_dashboard, name="dashboard"),
]
