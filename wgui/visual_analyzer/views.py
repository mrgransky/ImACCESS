import os
import random
import urllib.request

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from visual_analyzer.src.mcr_backend import generate_mcr
from visual_analyzer.src.txt_2_topk_imgs_backend import get_topkIMGs
from visual_analyzer.src.img_2_txt_backend import generate_caption, generate_labels

rand_extension = random.randint(0, 9999)

def intro_page(request):
	context = {
		'project_title': "ImACCESS",
		'greeting_text': "Hi there Researcher",
	}
	return render(request, 'visual_analyzer/intro_page.html', context)

def instruction_page(request):
	return render(request, 'visual_analyzer/instruction_page.html')

def about_page(request):
	return render(request, 'visual_analyzer/about_page.html')

def img2txt_page(request):
	context = {
		'project_title': "ImACCESS",
		'subject': 'Image-to-Text Retrieval',
	}
	if request.method == 'POST' and (request.FILES.get('image_file') or request.POST.get('image_url')):
		print(f"Uploaded image: ", end="\t")
		image_file = None
		if request.FILES.get('image_file'):
			print(f"FILE")
			image_file = request.FILES['image_file']
			fs = FileSystemStorage()
			file_path = fs.save(image_file.name, image_file)
			full_image_path = fs.path(file_path)
		elif request.POST.get('image_url'):
			print(f"URL")
			full_image_path = request.POST.get('image_url')  # Implement this function
		print(f">> input full_image_path: {full_image_path} | {type(full_image_path)}")
		if full_image_path:
			labels = generate_labels(
				img_source=full_image_path, 
				# rnd=rand_extension, 
				backend_method="clip",
			)
			caption = generate_caption(
				img_source=full_image_path, 
				# rnd=rand_extension, 
				backend_method="blip",
			)
			mcr_img_base64 = generate_mcr(
				img_source=full_image_path, 
				# rnd=rand_extension,
			)
			context['caption'] = caption
			context['lbls'] = labels
			context['result_image'] = mcr_img_base64
	return render(request, 'visual_analyzer/img2txt_page.html', context)

def txt2img_page(request):
	context = {
		'project_title': "ImACCESS",
		'subject': 'Text-to-Image Retrieval',
		'topk': 5,
	}
	if request.method == 'POST' and request.POST.get('user_text'):
		user_text = request.POST.get('user_text')
		print(f"Handling User Query Prompt: {user_text}...")
		context['user_query_prompt'] = user_text
		topk_images = get_topkIMGs(query=user_text)
		context['topK_imgs'] = topk_images
	return render(request, 'visual_analyzer/txt2img_page.html', context)

def img2img_page(request):
	context = {
		'project_title': "ImACCESS",
		'subject': 'Image-to-Image Retrieval',
	}
	return render(request, 'visual_analyzer/img2img_page.html', context)