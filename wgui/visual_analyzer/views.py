# from django.shortcuts import render
# from django.http import HttpResponse

# def main_page(request):
# 	if request.method == 'POST':
# 		# Check which input type was selected (URL or file)
# 		input_type = request.POST.get('input_type')
# 		# Handle URL input
# 		if input_type == 'url':
# 			image_url = request.POST.get('image_url')  # Get the URL from the form
# 			print(f"Image URL: {image_url}")  # Print to console for debugging purposes
# 			context = {
# 				'welcome_text': f"URL submitted: {image_url}",
# 			}
# 		# Handle file upload
# 		elif input_type == 'file':
# 			if 'image_file' in request.FILES:
# 				image_file = request.FILES['image_file']  # Get the uploaded file
# 				print(f"Uploaded File Path: {image_file.name}")  # Print the file name for debugging purposes
# 				context = {
# 					'welcome_text': f"File uploaded: {image_file.name}",
# 				}
# 			else:
# 				context = {
# 					'welcome_text': "No file uploaded.",
# 				}
# 	else:
# 		context = {
# 			'welcome_text': "Welcome to My System!",
# 		}
# 	return render(request, 'visual_analyzer/main_page.html', context)

import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from visual_analyzer.src.img_cap_backend import generate_caption
from visual_analyzer.src.mcr_backend import generate_mcr

import urllib.request

def main_page(request):
	context = {'welcome_text': "Welcome to My System!"}
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
			caption = generate_caption(img_source=full_image_path)
			mcr_img_base64 = generate_mcr(img_source=full_image_path)
			context['caption'] = caption
			context['result_image'] = mcr_img_base64
	return render(request, 'visual_analyzer/main_page.html', context)