#!/bin/bash

## How to run:
## $ nohup bash mcr_pouta.sh > /dev/null 2>&1 &
## $ nohup bash mcr_pouta.sh > /media/volume/trash/IMG/mcr.out 2>&1 & # with output saved in logs.out

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
echo "HOME DIR $HOME_DIR"
source $HOME_DIR/miniconda3/bin/activate py39
python -u mcr_new.py