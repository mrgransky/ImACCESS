#!/bin/bash

## run using command:
## $ nohup bash imaccess_wgui_pouta.sh > /dev/null 2>&1 &
## $ nohup bash imaccess_wgui_pouta.sh >> /media/volume/trash/ImACCESS/website_logs.out 2>&1 & # with output saved in logs.out

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
echo "HOME DIR $HOME_DIR"
source $HOME_DIR/miniconda3/bin/activate py39
rm -rf staticfiles; python manage.py collectstatic;
python -u manage.py runserver 0.0.0.0:9000