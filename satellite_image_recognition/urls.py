from django.conf.urls import url
from django.conf.urls.static import static

from . import views
from . import settings

urlpatterns = [
    url(r'upload/$', views.upload, name='upload'),
    url(r'results/$', views.results, name='results'),
    url(r'map/$', views.map, name='map'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
