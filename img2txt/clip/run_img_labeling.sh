#!/bin/bash

# Input argument(s):
img_pth=$1  # path to input image
out_pth=$2  # path to save output caption

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Bash job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
HOME_DIR=$(echo $HOME)
echo "HOME_DIR: $HOME_DIR | img: $img_pth | output: $out_pth"
python get_labels.py --image_path "$img_pth" --output_path "$out_pth"