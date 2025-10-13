from utils import *

img_path = "/home/farid/datasets/TEST_IMGs/baseball.jpeg"

IMG_MAX_RES = 512  # or 384
img = Image.open(img_path).convert("RGB")
print(f"Original image size: {img.size} => {np.array(img).shape} {np.array(img).dtype} {np.array(img).min()}/{np.array(img).max()} mode: {img.mode}")

# Create a copy for thumbnail (preserves aspect ratio)
img_thumbnail = img.copy()
img_thumbnail.thumbnail((IMG_MAX_RES, IMG_MAX_RES), Image.LANCZOS)
img_thumbnail.save(f"test_{IMG_MAX_RES}_thumbnail.jpg")
print(f"Thumbnail image size: {img_thumbnail.size}")

# Calculate new size preserving aspect ratio
original_width, original_height = img.size
aspect_ratio = original_width / original_height

if original_width > original_height:
    new_width = IMG_MAX_RES
    new_height = int(IMG_MAX_RES / aspect_ratio)
else:
    new_height = IMG_MAX_RES
    new_width = int(IMG_MAX_RES * aspect_ratio)

img_resize_aspect = img.resize((new_width, new_height), Image.LANCZOS)
img_resize_aspect.save(f"test_{IMG_MAX_RES}_resize_aspect.jpg")
print(f"Aspect-preserved resize: {img_resize_aspect.size}")