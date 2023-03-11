#!/bin/sh

# execute in parallel
nohup python ./bin/arima.py --dataset weinstock > ./output/track_arima_weinstock.txt &
nohup python ./bin/arima.py --dataset colas > ./output/track_arima_colas.txt &
nohup python ./bin/arima.py --dataset dubosson > ./output/track_arima_dubosson.txt &
nohup python ./bin/arima.py --dataset hall > ./output/track_arima_hall.txt &
nohup python ./bin/arima.py --dataset iglu > ./output/track_arima_iglu.txt &

