#!/bin/bash

# Parameters
img_pth=$1  # path to input image
out_pth=$2  # path to save output labels

echo "$img_pth"
echo "$out_pth"

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Bash job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
echo "HOME DIR $HOME_DIR"
source $HOME_DIR/miniconda3/bin/activate py39

python get_labels.py --image_path "$img_pth" --output_path "$out_pth"