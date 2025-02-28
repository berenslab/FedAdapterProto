#!/bin/bash
# bash ./scripts/baseline.sh
echo script name: $0

$TRAIN_DIR=...

python ${TRAIN_DIR}/federated_main.py \
    --mode task_heter \
    --dataset fundus \
    --num_classes 2 \
    --num_users 4 \
    --rounds 50 \
    --local_bs 16 \
    --num_channels 3 \
    --lr 0.0001 \
    --optimizer adam \
    --train_ep 1 \
    --use_sampler 1 \
    --model FedAdapterPrototype \
    --log_name fedAdapProtoProxLiActFTsubsetinit.log \
    --img_size 224 \
    --adapt_agg fednova_weighted_agg \
    --init_weight_proto 1 \
    --init_weight_adapter 1