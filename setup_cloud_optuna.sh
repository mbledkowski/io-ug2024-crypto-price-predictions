#!/bin/sh
git lfs fetch --all
git lfs pull

pyenv install 3.12.8
pyenv local 3.12.8
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

kaggle datasets download tencars/392-crypto-currency-pairs-at-minute-resolution
unzip 392-crypto-currency-pairs-at-minute-resolution.zip -d ./data/
(cd data && ls | grep -oP "^((?!.*[^-_]usd.csv).)*" | xargs -n1 rm)
(cd data && ls | grep -oP ".*test.*" | xargs -n1 rm)

# mysql -u root -e "CREATE DATABASE IF NOT EXISTS optuna"
# optuna create-study --study-name "distributed-optuna" --directions minimize minimize --storage "mysql+pymysql://root@localhost/optuna"
