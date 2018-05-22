#!/bin/sh

python3 retrain_last_layer.py -t ../Data/Training/Gros\ plan/ ../Data/Training/Plan\ rapproche/ ../Data/Training/Plan\ moyen/ -v ../Data/Test/Gros\ plan/ ../Data/Test/Plan\ rapproche/ ../Data/Test/Plan\ rapproche/ -transform log1p -lr 0.0001 -e 40 -csv out_log1p_0.0001lr.csv
python3 retrain_last_layer.py -t ../Data/Training/Gros\ plan/ ../Data/Training/Plan\ rapproche/ ../Data/Training/Plan\ moyen/ -v ../Data/Test/Gros\ plan/ ../Data/Test/Plan\ rapproche/ ../Data/Test/Plan\ rapproche/ -transform tanh -lr 0.0001 -e 40 -csv out_tanh_0.0001lr.csv

python3 retrain_last_layer.py -t ../Data/Training/Gros\ plan/ ../Data/Training/Plan\ rapproche/ ../Data/Training/Plan\ moyen/ -v ../Data/Test/Gros\ plan/ ../Data/Test/Plan\ rapproche/ ../Data/Test/Plan\ rapproche/ -transform tanh -lr 0.001 -e 40 -csv out_tanh_0.001lr.csv
python3 retrain_last_layer.py -t ../Data/Training/Gros\ plan/ ../Data/Training/Plan\ rapproche/ ../Data/Training/Plan\ moyen/ -v ../Data/Test/Gros\ plan/ ../Data/Test/Plan\ rapproche/ ../Data/Test/Plan\ rapproche/ -transform log1p -lr 0.001 -e 40 -csv out_log1p_0.001lr.csv

python3 retrain_last_layer.py -t ../Data/Training/Gros\ plan/ ../Data/Training/Plan\ rapproche/ ../Data/Training/Plan\ moyen/ -v ../Data/Test/Gros\ plan/ ../Data/Test/Plan\ rapproche/ ../Data/Test/Plan\ rapproche/ -transform tanh -lr 0.01 -e 40 -csv out_tanh_0.01lr.csv
python3 retrain_last_layer.py -t ../Data/Training/Gros\ plan/ ../Data/Training/Plan\ rapproche/ ../Data/Training/Plan\ moyen/ -v ../Data/Test/Gros\ plan/ ../Data/Test/Plan\ rapproche/ ../Data/Test/Plan\ rapproche/ -transform log1p -lr 0.01 -e 40 -csv out_log1p_0.01lr.csv
