#!/usr/bin/env bash
#install venv on UCloud:
sudo apt-get update
sudo apt-get install python3-venv
#create virtual environment
python3 -m venv spacy_env
# activate virtual environment
source ./spacy_env/bin/activate

# then install requirements
python 3 -m pip install ---upgrade pip
python3 -m pip install -r requirements.txt
python3 spacy download en_core_web_md

#run
python3 src/script.py

# deactivate the venv
deactivate