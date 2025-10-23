import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(parent_dir)
print(project_dir)
print(os.listdir(project_dir))
sys.path.insert(0, project_dir) # add project directory to sys.path
from misc.utils import *
from misc.visualize import *
from mediapipe.tasks import python

language_detector = "language_detector.tflite"
if language_detector not in os.listdir():
	url = f"https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/{language_detector}"
	urllib.request.urlretrieve(url, language_detector)

input_text = "Golda Meir resigns as prime minister."

print("Running mediapipe Language Detector on CPU...")
base_options = python.BaseOptions(model_asset_path=language_detector)
options = python.text.LanguageDetectorOptions(base_options=base_options)
with python.text.LanguageDetector.create_from_options(options) as detector:
	detection_result = detector.detect(input_text)
	print("\nLanguage Detection Result:")
	print(detection_result)

language_classifier = "bert_classifier.tflite"
if language_classifier not in os.listdir():
	url = f"https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/1/{language_classifier}"
	urllib.request.urlretrieve(url, language_classifier)

print("\nRunning Text Classifier on CPU...")
base_options = python.BaseOptions(model_asset_path=language_classifier)
options = python.text.TextClassifierOptions(base_options=base_options)

with python.text.TextClassifier.create_from_options(options) as classifier:
	classification_result = classifier.classify(input_text)
	print("\nText Classification Result:")
	top_category = classification_result.classifications[0].categories[0]
	print(f'{top_category.category_name} ({top_category.score:.2f})')