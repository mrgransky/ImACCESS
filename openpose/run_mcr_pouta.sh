#!/bin/bash

## How to run:
## $ nohup bash run_mcr_pouta.sh > /dev/null 2>&1 &
## $ nohup bash run_mcr_pouta.sh >> /media/volume/trash/IMGs/results.out 2>&1 & # with output saved in logs.out

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
echo "HOME DIR $HOME_DIR"
source $HOME_DIR/miniconda3/bin/activate py39

# it's supposed to work for seveal images in a directory,
# but it get's stuck due to insufficient GPU. 
# therefore, media_x1 contains only 1 image in a directory 
python -u mcr_orig.py --imgs_dir examples/media_x1