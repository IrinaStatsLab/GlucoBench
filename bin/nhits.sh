#!/bin/sh

# execute in parallel
nohup python ./lib/nhits.py --dataset weinstock --use_covs False --optuna False > ./output/track_nhits_weinstock.txt &
nohup python ./lib/nhits.py --dataset weinstock --use_covs True --optuna False > ./output/track_nhits_covariates_weinstock.txt &
nohup python ./lib/nhits.py --dataset colas --use_covs False --optuna False > ./output/track_nhits_colas.txt &
nohup python ./lib/nhits.py --dataset colas --use_covs True --optuna False > ./output/track_nhits_covariates_colas.txt &
nohup python ./lib/nhits.py --dataset dubosson --use_covs False --optuna False > ./output/track_nhits_dubosson.txt &
nohup python ./lib/nhits.py --dataset dubosson --use_covs True --optuna False > ./output/track_nhits_covariates_dubosson.txt &
nohup python ./lib/nhits.py --dataset hall --use_covs False --optuna False > ./output/track_nhits_hall.txt &
nohup python ./lib/nhits.py --dataset hall --use_covs True --optuna False > ./output/track_nhits_covariates_hall.txt &
nohup python ./lib/nhits.py --dataset iglu --use_covs False --optuna False > ./output/track_nhits_iglu.txt &
nohup python ./lib/nhits.py --dataset iglu --use_covs True --optuna False > ./output/track_nhits_covariates_iglu.txt &
