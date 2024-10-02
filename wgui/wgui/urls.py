from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static
from visual_analyzer.views import intro_page, txt2img_page, img2txt_page, img2img_page

urlpatterns = [
	path('admin/', admin.site.urls),
	path('img2txt/', img2txt_page, name='img2txt'),
	path('txt2img/', txt2img_page, name='txt2img'),
	path('img2img/', img2img_page, name='img2img'),
	path('', intro_page),  # Map root URL to intro page
]

if settings.DEBUG:
	import debug_toolbar
	urlpatterns = [
		path('__debug__/', include(debug_toolbar.urls)),
	] + urlpatterns
	
	# Serve media files during development
	urlpatterns += static(
		settings.MEDIA_URL,
		document_root=settings.MEDIA_ROOT,
	)