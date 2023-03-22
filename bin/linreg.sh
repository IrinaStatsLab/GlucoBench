#!/bin/sh

# execute in parallel
nohup python ./lib/linreg.py --dataset weinstock --use_covs False --optuna True > ./output/track_linreg_weinstock.txt &
nohup python ./lib/linreg.py --dataset weinstock --use_covs True --optuna True > ./output/track_linreg_covariates_weinstock.txt &
nohup python ./lib/linreg.py --dataset colas --use_covs False --optuna True > ./output/track_linreg_colas.txt &
nohup python ./lib/linreg.py --dataset colas --use_covs True --optuna True > ./output/track_linreg_covariates_colas.txt &
nohup python ./lib/linreg.py --dataset dubosson --use_covs False --optuna True > ./output/track_linreg_dubosson.txt &
nohup python ./lib/linreg.py --dataset dubosson --use_covs True --optuna True > ./output/track_linreg_covariates_dubosson.txt &
nohup python ./lib/linreg.py --dataset hall --use_covs False --optuna True > ./output/track_linreg_hall.txt &
nohup python ./lib/linreg.py --dataset hall --use_covs True --optuna True > ./output/track_linreg_covariates_hall.txt &
nohup python ./lib/linreg.py --dataset iglu --use_covs False --optuna True > ./output/track_linreg_iglu.txt &
nohup python ./lib/linreg.py --dataset iglu --use_covs True --optuna True > ./output/track_linreg_covariates_iglu.txt &
