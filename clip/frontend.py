from backend import run_backend
import datetime
import time
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random

from torchvision.datasets import ImageNet
train_dataset = ImageNet(
	root=os.path.expanduser("~/.cache"),
	split='train',
	download=True,
	transform=None
)
test_dataset = ImageNet(
	root=os.path.expanduser("~/.cache"),
	split='val',
	download=True,
	transform=None
)

class CINIC10(Dataset):
	classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	def __init__(self, root, train=True, download=False, transform=None):
		self.root = root
		self.train = train
		self.transform = transform
		if train:
			self.data = self._load_data(os.path.join(root, 'train'))
		else:
			self.data = self._load_data(os.path.join(root, 'valid'))

	def _load_data(self, directory):
		data = []
		labels = []
		for idx, class_name in enumerate(self.classes):
			# print(f"Loading {idx} {class_name} images...")
			class_dir = os.path.join(directory, class_name)
			for file_name in os.listdir(class_dir):
				file_path = os.path.join(class_dir, file_name)
				data.append(file_path)
				labels.append(idx)
		return list(zip(data, labels))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		file_path, label = self.data[index]
		image = Image.open(file_path)
		if self.transform is not None:
			image = self.transform(image)
		return image, label

	def __repr__(self):
		split = 'Train' if self.train else 'Test'
		return (
			f'Dataset CINIC10\n' \
			f'    Number of datapoints: {len(self)}\n' \
			f'    Root location: {self.root}\n' \
			f'    Split: {split}'
		)

import matplotlib.pyplot as plt

cinic10_dataset_train = CINIC10(root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/CINIC-10', train=True)
random_indices = random.sample(range(len(cinic10_dataset_train)), 10)
print(random_indices)
for i in random_indices:
		image, label = cinic10_dataset_train[i]
		print(f"Image path: {cinic10_dataset_train.data[i][0]}")
		print(f"Label: {label} ({cinic10_dataset_train.classes[label]})")
		print("-" * 50)

# Visualize some images and their corresponding labels
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i, idx in enumerate(random_indices):
		image, label = cinic10_dataset_train[idx]
		axs[i // 5, i % 5].imshow(image)
		axs[i // 5, i % 5].set_title(cinic10_dataset_train.classes[label])
		axs[i // 5, i % 5].axis('off')
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