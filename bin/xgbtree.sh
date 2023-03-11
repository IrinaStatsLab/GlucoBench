#!/bin/sh

# execute in parallel
nohup python ./bin/xgbtree.py --dataset weinstock --use_covs False --optuna False > ./output/track_xgboost_weinstock.txt &
nohup python ./bin/xgbtree.py --dataset weinstock --use_covs True --optuna False > ./output/track_xgboost_covariates_weinstock.txt &
nohup python ./bin/xgbtree.py --dataset colas --use_covs False --optuna False > ./output/track_xgboost_colas.txt &
nohup python ./bin/xgbtree.py --dataset colas --use_covs True --optuna False > ./output/track_xgboost_covariates_colas.txt &
nohup python ./bin/xgbtree.py --dataset dubosson --use_covs False --optuna False > ./output/track_xgboost_dubosson.txt &
nohup python ./bin/xgbtree.py --dataset dubosson --use_covs True --optuna False > ./output/track_xgboost_covariates_dubosson.txt &
nohup python ./bin/xgbtree.py --dataset hall --use_covs False --optuna False > ./output/track_xgboost_hall.txt &
nohup python ./bin/xgbtree.py --dataset hall --use_covs True --optuna False > ./output/track_xgboost_covariates_hall.txt &
nohup python ./bin/xgbtree.py --dataset iglu --use_covs False --optuna False > ./output/track_xgboost_iglu.txt &
nohup python ./bin/xgbtree.py --dataset iglu --use_covs True --optuna False > ./output/track_xgboost_covariates_iglu.txt &
