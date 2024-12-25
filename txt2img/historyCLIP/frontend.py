from backend import run_backend
import datetime
import time

def main():
  run_backend()
  return

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	START_EXECUTION_TIME = time.time()
	main()
	END_EXECUTION_TIME = time.time()
	print(
		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
		.center(160, " ")
	)