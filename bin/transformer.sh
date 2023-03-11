#!/bin/sh

# execute in parallel
# nohup python ./bin/transformer.py --dataset weinstock --use_covs False --optuna False > ./output/track_transformer_weinstock.txt &
nohup python ./bin/transformer.py --dataset weinstock --use_covs True --optuna False > ./output/track_transformer_covariates_weinstock.txt &
# nohup python ./bin/transformer.py --dataset colas --use_covs False --optuna False > ./output/track_transformer_colas.txt &
nohup python ./bin/transformer.py --dataset colas --use_covs True --optuna False > ./output/track_transformer_covariates_colas.txt &
# nohup python ./bin/transformer.py --dataset dubosson --use_covs False --optuna False > ./output/track_transformer_dubosson.txt &
nohup python ./bin/transformer.py --dataset dubosson --use_covs True --optuna False > ./output/track_transformer_covariates_dubosson.txt &
# nohup python ./bin/transformer.py --dataset hall --use_covs False --optuna False > ./output/track_transformer_hall.txt &
nohup python ./bin/transformer.py --dataset hall --use_covs True --optuna False > ./output/track_transformer_covariates_hall.txt &
# nohup python ./bin/transformer.py --dataset iglu --use_covs False --optuna False > ./output/track_transformer_iglu.txt &
nohup python ./bin/transformer.py --dataset iglu --use_covs True --optuna False > ./output/track_transformer_covariates_iglu.txt &
