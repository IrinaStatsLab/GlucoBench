#!/bin/sh

# execute in parallel
nohup python ./lib/gluformer.py --dataset weinstock --gpu_id 1 --optuna False > ./output/track_gluformer_weinstock.txt &
# nohup python ./lib/gluformer.py --dataset colas --gpu_id 2 --optuna False > ./output/track_gluformer_colas.txt &
# nohup python ./lib/gluformer.py --dataset dubosson --gpu_id 0 --optuna False > ./output/track_gluformer_dubosson.txt &
# nohup python ./lib/gluformer.py --dataset hall --gpu_id 3 --optuna False > ./output/track_gluformer_hall.txt &
# nohup python ./lib/gluformer.py --dataset iglu --gpu_id 0 --optuna False > ./output/track_gluformer_iglu.txt &

