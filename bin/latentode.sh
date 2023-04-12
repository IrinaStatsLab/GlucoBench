#!/bin/sh

# execute in parallel
nohup python ./lib/latentode.py --dataset weinstock --gpu_id 3 --optuna False > ./output/track_latentode_weinstock.txt &
nohup python ./lib/latentode.py --dataset colas --gpu_id 2 --optuna False > ./output/track_latentode_colas.txt &
nohup python ./lib/latentode.py --dataset dubosson --gpu_id 0 --optuna False > ./output/track_latentode_dubosson.txt &
nohup python ./lib/latentode.py --dataset hall --gpu_id 1 --optuna False > ./output/track_latentode_hall.txt &
nohup python ./lib/latentode.py --dataset iglu --gpu_id 0 --optuna False > ./output/track_latentode_iglu.txt &

