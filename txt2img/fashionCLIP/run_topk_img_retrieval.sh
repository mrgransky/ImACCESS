#!/bin/bash

# Input argument(s):
query_prompt=$1  # input Query Prompt
out_pth=$2  # path to save output TOP-K images (concatinated in one!)

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Bash job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
HOME_DIR=$(echo $HOME)
echo "HOME_DIR: $HOME_DIR | Query: $query_prompt | output: $out_pth"
python topk_image_retrieval.py --query "$query_prompt" --processed_image_path "$out_pth"