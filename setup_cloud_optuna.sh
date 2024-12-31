./setup.sh
mysql -u root -e "CREATE DATABASE IF NOT EXISTS optuna"
optuna create-study --study-name "distributed-optuna" --storage "mysql://root@localhost/optuna"
