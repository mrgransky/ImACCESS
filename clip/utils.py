import os
import torch
import clip
import time
import re
import argparse
import random
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
import warnings
import datetime
warnings.filterwarnings("ignore", category=UserWarning, module='torch.optim.lr_scheduler')

def set_seeds(seed:int=42, debug:bool=False):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		if debug: # slows down training but ensures reproducibility
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False