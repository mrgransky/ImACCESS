import os
import pandas as pd
import datetime
import time
from utils import *

def get_A():
  print("A")
  
def get_B():
  print("B")
  
def get_plot():
	print("plotting fcn...")
	plot_loss_accuracy(
		train_losses=[1, 2, 3, 4, 5],
		val_losses=[1, 2, 3, 4, 5],
		validation_accuracy_text_description_for_each_image_list=[1, 2, 3, 4, 5],
		validation_accuracy_text_image_for_each_text_description_list=[1, 2, 3, 4, 5],
		dataset_name="CIFAR10",
		learning_rate=1e-5,
		weight_decay=1e-3,
		batch_size=64,
	)
    
def get_A_B_C():
	get_A()
	get_B()
	get_plot()
	return

def run_backend():
	get_A_B_C()
	return

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	START_EXECUTION_TIME = time.time()
	run_backend()
	END_EXECUTION_TIME = time.time()
	print(
		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
		.center(160, " ")
	)