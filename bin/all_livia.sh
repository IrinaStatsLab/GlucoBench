#!/usr/bin/env sh

#these still keep the seeds
python3 ./lib/linreg.py --dataset livia --use_covs False --optuna False > ./output/track_linreg_livia.txt
python3 ./lib/xgbtree.py --dataset livia --use_covs False --optuna False > ./output/track_xgboost_livia.txt 


#these are original versions with seeds
#python3 ./lib/nhits.py --dataset livia --use_covs False --optuna False > ./output/track_nhits_livia.txt 
#pickle waiting
#python3 ./lib/latentode.py --dataset livia --gpu_id 0 --optuna False > ./output/track_latentode_livia.txt
#wanted to run with cuda but couldn't?????
#python3 ./lib/transformer.py --dataset livia --use_covs False --optuna False > ./output/track_transformer_livia.txt
# no gpu enableing found
#python3 ./lib/gluformer.py --dataset livia --gpu_id 0 --optuna False > ./output/track_gluformer_livia.txt
##no module torch.-six



#these are seedless
python3 ./lib/nhits_2.py --dataset livia --use_covs False > ./output/track_nhits_livia.txt 
python3 ./lib/latentode_2.py --dataset livia --gpu_id 0 > ./output/track_latentode_livia.txt
python3 ./lib/transformer_2.py --dataset livia --use_covs False > ./output/track_transformer_livia.txt
python3 ./lib/gluformer_2.py --dataset livia --gpu_id 0 > ./output/track_gluformer_livia.txt




#these i haven't checked
python3 ./lib/tft.py --dataset livia --use_covs False --optuna False > ./output/track_tft_livia.txt
##not wnough value to unpack- expected 6 got 5
python3 ./lib/arima.py --dataset livia > ./output/track_livia.txt
## issue with model declaration, and some no errors found that can't be concat 