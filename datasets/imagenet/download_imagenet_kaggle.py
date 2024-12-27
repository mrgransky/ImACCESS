import json
import os
from pathlib import Path
import subprocess
import time
import argparse

# local:
# $ nohup python -u download_imagenet_kaggle.py --save_dir /home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs > logs/download_log.out &


# Pouta:
# $ nohup python -u download_imagenet_kaggle.py --save_dir /media/volume/ImACCESS/WW_DATASETs/IMAGENET > logs/download_log.out &

def setup_kaggle_credentials():
		"""Setup Kaggle credentials from json file"""
		print("Setting up Kaggle credentials...")
		
		kaggle_dir = Path.home() / '.kaggle'
		print(f"Checking Kaggle directory: {kaggle_dir}")
		
		if not kaggle_dir.exists():
				print("Creating .kaggle directory...")
				kaggle_dir.mkdir(exist_ok=True)
				print("Directory created successfully")
		else:
				print(".kaggle directory already exists")

		credentials = {
				"username": "mrgrandsky",
				"key": "54a892faa42231616d359de6ad596351"
		}
		
		kaggle_cred_path = kaggle_dir / 'kaggle.json'
		print(f"Writing credentials to: {kaggle_cred_path}")
		
		with open(kaggle_cred_path, 'w') as f:
				json.dump(credentials, f)
		
		os.chmod(kaggle_cred_path, 0o600)
		print("Credentials file permissions set to 600")
		print("Kaggle credentials setup completed\n")

def download_dataset(save_dir: Path):
		"""Download the ImageNet dataset using kaggle command"""
		print("Initiating dataset download...")
		start_time = time.time()
		
		# Create save directory if it doesn't exist
		save_dir.mkdir(parents=True, exist_ok=True)
		print(f"Download directory: {save_dir}")
		
		try:
				print("Executing Kaggle download command...")
				command = ["kaggle", "competitions", "download", "-c", "imagenet-object-localization-challenge", "-p", str(save_dir)]
				
				# Run the command with real-time output
				process = subprocess.Popen(
						command,
						stdout=subprocess.PIPE,
						stderr=subprocess.PIPE,
						universal_newlines=True,
						bufsize=1
				)
				
				# Print output in real-time
				while True:
						output = process.stdout.readline()
						if output == '' and process.poll() is not None:
								break
						if output:
								print(f"Progress: {output.strip()}")
								
				# Get the return code
				return_code = process.poll()
				
				if return_code == 0:
						elapsed_time = time.time() - start_time
						print(f"\nDownload completed successfully!")
						print(f"Total download time: {elapsed_time:.2f} seconds")
						
						# Check downloaded file size
						for file in save_dir.glob("*.zip"):
								size_gb = file.stat().st_size / (1024 * 1024 * 1024)  # Convert to GB
								print(f"Downloaded file: {file.name} (Size: {size_gb:.2f} GB)")
								
						return True
				else:
						error = process.stderr.read()
						print(f"Download failed with error:\n{error}")
						return False
						
		except Exception as e:
				print(f"An unexpected error occurred: {str(e)}")
				return False

def parse_args():
		parser = argparse.ArgumentParser(description='Download ImageNet dataset from Kaggle')
		parser.add_argument('--save_dir', type=str, default='./imagenet_data',
												help='Directory to save the downloaded dataset (default: ./imagenet_data)')
		return parser.parse_args()

if __name__ == "__main__":
		print("=== Starting Kaggle Dataset Download Script ===\n")
		
		# Parse command line arguments
		args = parse_args()
		save_dir = Path(args.save_dir)
		
		try:
				# Check if kaggle is installed
				subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
				print("Kaggle CLI is installed and accessible\n")
		except subprocess.CalledProcessError:
				print("Error: Kaggle CLI not found. Please install it using: pip install kaggle")
				exit(1)
		except FileNotFoundError:
				print("Error: Kaggle CLI not found. Please install it using: pip install kaggle")
				exit(1)
				
		setup_kaggle_credentials()
		success = download_dataset(save_dir)
		
		if success:
				print(f"\nScript completed successfully! Dataset saved in: {save_dir}")
		else:
				print("\nScript completed with errors.")