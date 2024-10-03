#!/bin/bash

# Input argument(s):
query_prompt=$1  # input Query Prompt
out_pth=$2  # path to save output TOP-K images (concatenated in one!)
user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Bash job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
HOME_DIR=$(echo $HOME)
USERNAME=$(echo $USER)

echo "HOME_DIR: $HOME_DIR | USER: $USERNAME | Query: $query_prompt | output: $out_pth"

# Set --num_epochs based on the USERNAME
if [ "$USERNAME" == "ubuntu" ]; then
		num_epochs=50
else
		num_epochs=3
fi

# Run the Python script with the appropriate --num_epochs value
python -u topk_image_retrieval.py \
		--query "$query_prompt" \
		--processed_image_path "$out_pth" \
		--num_epochs $num_epochs

echo "Done!"