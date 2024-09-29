import os
import random

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from visual_analyzer.src.img_cap_backend import generate_caption
from visual_analyzer.src.mcr_backend import generate_mcr
from visual_analyzer.src.img_lbls_backend import generate_labels

import urllib.request

def main_page(request):
	# context = {'welcome_text': "Welcome to ImACCESS"}
	context = {
		'project_title': "ImACCESS",
		'greeting_text': "Hi there Researcher",
	}
	if request.method == 'POST' and (request.FILES.get('image_file') or request.POST.get('image_url')):
		print(f"Handling uploaded image...")
		# Handle file upload
		image_file = None
		if request.FILES.get('image_file'):
			image_file = request.FILES['image_file']
			fs = FileSystemStorage()
			file_path = fs.save(image_file.name, image_file)
			full_image_path = fs.path(file_path)
		elif request.POST.get('image_url'):
			full_image_path = request.POST.get('image_url')  # Implement this function
		print(f">> input full_image_path: {full_image_path} | {type(full_image_path)}")
		if full_image_path:
			rand_extension = random.randint(0, 9999)
			labels = generate_labels(img_source=full_image_path, rnd=rand_extension, backend_method="clip")
			caption = generate_caption(img_source=full_image_path, rnd=rand_extension, backend_method="blip")
			mcr_img_base64 = generate_mcr(img_source=full_image_path, rnd=rand_extension)
			context['lbls'] = labels
			context['caption'] = caption
			context['result_image'] = mcr_img_base64

	return render(request, 'visual_analyzer/main_page.html', context)