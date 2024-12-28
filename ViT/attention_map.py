import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTFeatureExtractor, ViTModel, ViTImageProcessor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Load a sample from CIFAR-10 dataset
transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

cifar10_dataset = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), train=True, download=True, transform=transform)
loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

# Load pre-trained ViT model and image processor
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_attentions=True)
model.eval()  # Set the model to evaluation mode

# Process one image
sample = next(iter(loader))
image_tensor = sample[0].squeeze(0)  # shape should now be [3, 32, 32]

# Convert normalized tensor back to image for visualization
image_for_display = image_tensor.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
image_for_display = (image_for_display + 1) / 2.0  # Undo normalization

# Prepare image for ViT model - convert tensor to PIL image
pil_image = transforms.ToPILImage()(image_tensor).convert("RGB")

# Use image processor on PIL image
inputs = image_processor(images=pil_image, return_tensors="pt")

with torch.no_grad():
		outputs = model(**inputs)

# Check if attentions are available
if outputs.attentions is not None:
		attention = outputs.attentions[-1][0].mean(dim=1)  # Average across heads
else:
		print("Attention weights are not available. Make sure 'output_attentions' is set to True.")
		exit()

# Resize attention map to match the input image dimensions
attention_map = attention[0, 1:].view(14, 14).cpu().numpy()  # Removing CLS token
attention_map = np.interp(attention_map, (attention_map.min(), attention_map.max()), (0, 1))
resized_attention_map = np.array(Image.fromarray(attention_map).resize((32, 32), Image.BICUBIC))

# Create heatmap
heatmap = plt.cm.jet(resized_attention_map)[:, :, :3]  # Exclude alpha channel

# Display results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))

# Original image
ax1.imshow(image_for_display)
ax1.set_title('Original Image')
ax1.axis('off')

# Attention map (grayscale)
ax2.imshow(resized_attention_map, cmap='viridis')
ax2.set_title('Attention Map')
ax2.axis('off')

# Heatmap
ax3.imshow(image_for_display)
ax3.imshow(heatmap, alpha=0.5)  # Overlay heatmap on the image
ax3.set_title('Attention Heatmap')
ax3.axis('off')

plt.tight_layout()
plt.savefig("attention_map.png")