#!/bin/bash
# bash ./scripts/baseline.sh
echo script name: $0

python exps/federated_main.py --mode task_heter --dataset mnist --num_classes 10 --num_users 20 --ways 5 --stdev 2 --rounds 200

python exps/federated_main.py --mode model_heter --dataset mnist --num_classes 10 --num_users 20 --ways 5 --stdev 2 --rounds 200

python exps/federated_main.py --mode task_heter --dataset cifar10 --num_classes 10 --rounds 30 --train_ep 8 --ways 4 --stdev 1

# my version
python federated_main.py --mode task_heter --dataset cifar10 --num_classes 10 --local_bs 8 --num_channels 3 --ld 0.1 --rounds 110 --ways 3
python federated_main.py --mode task_heter --dataset cifar10 --num_classes 10 --local_bs 32 --num_channels 3 --ld 0.1 --rounds 110 --ways 3 --num_users 20 --shot 100 --stdev 1
python federated_main.py --mode task_heter --dataset oct --num_classes 2 --num_users 4 --rounds 1 --local_bs 32 --num_channels 3 --lr 0.0001 --optimizer adam --train_ep 1