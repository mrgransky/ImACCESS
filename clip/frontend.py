from backend import run_backend
import datetime
import time
import os
import json
import random
import matplotlib.pyplot as plt
from datasets import CINIC10, ImageNet

# from torchvision.datasets import ImageNet
# train_dataset = ImageNet(
# 	root=os.path.expanduser("~/.cache"),
# 	split='train',
# 	download=True,
# 	transform=None
# )
# test_dataset = ImageNet(
# 	root=os.path.expanduser("~/.cache"),
# 	split='val',
# 	download=True,
# 	transform=None
# )

# cinic10_dataset_train = CINIC10(root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/CINIC10', train=True)
# random_indices = random.sample(range(len(cinic10_dataset_train)), 10)
# print(len(random_indices), type(random_indices), random_indices)
# for i in random_indices:
# 	image, label = cinic10_dataset_train[i]
# 	print(f"Image path: {cinic10_dataset_train.data[i][0]}")
# 	print(f"Label: {label} ({cinic10_dataset_train.classes[label]})")
# 	print("-" * 50)
# fig, axs = plt.subplots(2, 5, figsize=(20, 8))
# for i, idx in enumerate(random_indices):
# 	image, label = cinic10_dataset_train[idx]
# 	axs[i // 5, i % 5].imshow(image)
# 	axs[i // 5, i % 5].set_title(cinic10_dataset_train.classes[label])
# 	axs[i // 5, i % 5].axis('off')
# plt.title(f"CINIC10 Dataset {len(random_indices)} Random Images")
# plt.tight_layout()
# plt.show()

imgnet = ImageNet(
	root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/IMAGENET', 
	train=True,
)
print(imgnet)
random_indices = random.sample(range(len(imgnet)), 10)
print(random_indices)
print(len(random_indices), type(random_indices), random_indices)
for i in random_indices:
	image, label_idx = imgnet[i]
	# print(f"Image path: {imgnet.data[i][0]}")
	print(f"Image path: {imgnet.data[i]}")
	print(f"[Label] index: {label_idx} SynsetID: {imgnet.synset_ids[label_idx]} ({imgnet.classes[label_idx]})")
	print("-" * 50)
fig, axs = plt.subplots(2, 5, figsize=(22, 8))

for i, idx in enumerate(random_indices):
	image, label_idx = imgnet[idx]
	axs[i // 5, i % 5].imshow(image)
	axs[i // 5, i % 5].set_title(f"{imgnet.synset_ids[label_idx]} {imgnet.classes[label_idx]}", fontsize=8)
	axs[i // 5, i % 5].axis('off')
plt.suptitle(f"IMAGENET {len(random_indices)} Random Images")
plt.tight_layout()
plt.show()

# def main():
#   run_backend()
#   return

# if __name__ == "__main__":
# 	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
# 	START_EXECUTION_TIME = time.time()
# 	main()
# 	END_EXECUTION_TIME = time.time()
# 	print(
# 		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
# 		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
# 		.center(160, " ")
# 	)