#!/bin/sh

# execute in parallel
nohup python ./lib/tft.py --dataset weinstock --use_covs False --optuna False > ./output/track_tft_weinstock.txt &
# nohup python ./lib/tft.py --dataset weinstock --use_covs True --optuna False > ./output/track_tft_covariates_weinstock.txt &
# nohup python ./lib/tft.py --dataset colas --use_covs False --optuna False > ./output/track_tft_colas.txt &
# nohup python ./lib/tft.py --dataset colas --use_covs True --optuna False > ./output/track_tft_covariates_colas.txt &
# nohup python ./lib/tft.py --dataset dubosson --use_covs False --optuna False > ./output/track_tft_dubosson.txt &
# nohup python ./lib/tft.py --dataset dubosson --use_covs True --optuna False > ./output/track_tft_covariates_dubosson.txt &
# nohup python ./lib/tft.py --dataset hall --use_covs False --optuna False > ./output/track_tft_hall.txt &
# nohup python ./lib/tft.py --dataset hall --use_covs True --optuna False > ./output/track_tft_covariates_hall.txt &
# nohup python ./lib/tft.py --dataset iglu --use_covs False --optuna False > ./output/track_tft_iglu.txt &
# nohup python ./lib/tft.py --dataset iglu --use_covs True --optuna False > ./output/track_tft_covariates_iglu.txt &
