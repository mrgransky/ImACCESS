import os
import pandas as pd
import datetime
import time

def get_A():
  print("A")
  
def get_B():
  print("B")
  
def get_C():
	print("C")
    
def get_A_B_C():
	get_A()
	get_B()
	get_C()
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