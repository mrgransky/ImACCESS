import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /home/farid/WS_Farid/ImACCESS/wgui

PROJECT_DIR = os.path.abspath(
  os.path.join(
    os.path.dirname(__file__), 
    os.pardir, os.pardir,
  )
) # /home/farid/WS_Farid/ImACCESS

# sys.path.append(PROJECT_DIR)

print(f">> settings.py BASE_DIR: {BASE_DIR}")
print(f">> settings.py PROJECT_DIR: {PROJECT_DIR}")

SECRET_KEY = os.getenv(
	'DJANGO_SECRET_KEY',
	'9e4@&tw46$l31)zrqe3wi+-slqm(ruvz&se0^%9#6(_w3ui!c0'
)

CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'

DEBUG = False # To observe that css style layout changes have already taken place! => must be when executed in Pouta!
# DEBUG = True # causes "Broken pipe from ('94.72.62.225', 53959)" error! # just for debugging purposes!
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
SECURE_SSL_REDIRECT = False

ALLOWED_HOSTS = ['*']
# ALLOWED_HOSTS = ['128.214.254.157', '10.130.15.24', '127.0.0.1', 'localhost',]  # Add the IP address of your VM # $lsof -i :8000

INSTALLED_APPS = [
	'channels',
	'django.contrib.admin',
	'django.contrib.auth',
	'django.contrib.contenttypes',
	'django.contrib.sessions',
	'django.contrib.messages',
	'django.contrib.staticfiles',
	'debug_toolbar',
	'visual_analyzer',
]

MIDDLEWARE = [
	'django.middleware.security.SecurityMiddleware',
	'django.contrib.sessions.middleware.SessionMiddleware',
	'django.middleware.common.CommonMiddleware',
	'django.middleware.csrf.CsrfViewMiddleware',
	'django.contrib.auth.middleware.AuthenticationMiddleware',
	'django.contrib.messages.middleware.MessageMiddleware',
	'django.middleware.clickjacking.XFrameOptionsMiddleware',
	'whitenoise.middleware.WhiteNoiseMiddleware',
	'debug_toolbar.middleware.DebugToolbarMiddleware',
]

ROOT_URLCONF = 'wgui.urls'

TEMPLATES = [
	{
		'BACKEND': 'django.template.backends.django.DjangoTemplates',
		'DIRS': [],
		'APP_DIRS': True,
		'OPTIONS': {
			'context_processors': [
				'django.template.context_processors.debug',
				'django.template.context_processors.request',
				'django.contrib.auth.context_processors.auth',
				'django.contrib.messages.context_processors.messages',
			],
		},
	},
]

WSGI_APPLICATION = 'wgui.wsgi.application'
ASGI_APPLICATION = 'wgui.asgi.application'
# Add Channels layers configuration if using Redis or other layer
# CHANNEL_LAYERS = {
#     "default": {
#         "BACKEND": "channels_redis.core.RedisChannelLayer",
#         "CONFIG": {
#             "hosts": [("127.0.0.1", 6379)],
#         },
#     },
# }
from . import database
DATABASES = {
	'default': database.config()
}

DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'
AUTH_PASSWORD_VALIDATORS = [
	{
		'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
	},
	{
		'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
	},
	{
		'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
	},
	{
		'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
	},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# STATIC_URL = '/static/'
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')] # Extra places for collectstatic to find static files

# MEDIA settings
# MEDIA_URL = '/media/'
MEDIA_URL = 'media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

INTERNAL_IPS = ['127.0.0.1']