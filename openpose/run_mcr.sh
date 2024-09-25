#!/bin/bash

# Parameters
img_pth=$1  # path to input image
out_pth=$2  # path to save output caption

# echo "$img_pth"
# echo "$out_pth"

# Run captioning model (for example, using a Python script)
# This assumes you have a Python script in the captions directory that generates the caption
user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Bash job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
echo "HOME DIR $HOME_DIR"
source $HOME_DIR/miniconda3/bin/activate py39

python mcr.py --image_path "$img_pth" --processed_image_path "$out_pth"