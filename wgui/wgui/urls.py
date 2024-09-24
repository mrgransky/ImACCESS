from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static
from visual_analyzer.views import main_page

urlpatterns = [
	path('admin/', admin.site.urls),
	path('home/', main_page, name='main_page'),
	path('', main_page),  # Map root URL to main_page
]

if settings.DEBUG:
	import debug_toolbar
	urlpatterns = [
		path('__debug__/', include(debug_toolbar.urls)),
	] + urlpatterns
	
	# Serve media files during development
	urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)