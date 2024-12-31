#!/bin/sh

pyenv local 3.12.8
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mysql -u root -e "CREATE DATABASE IF NOT EXISTS optuna"
optuna create-study --study-name "distributed-optuna" --directions minimize minimize --storage "mysql://root@localhost/optuna"
