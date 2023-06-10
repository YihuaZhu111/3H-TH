#!/bin/bash

# k=32, WN18RR
python run.py \
            --dataset WN18RR \
            --model ThreeH_TH \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 500 \
            --patience 15 \
            --valid 5 \
            --batch_size 500 \
            --neg_sample_size 100 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg \
            --multi_c 

# k=32, FB237
python run.py \
            --dataset FB237 \
            --model ThreeH_TH \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adagrad \
            --max_epochs 500 \
            --patience 15 \
            --valid 5 \
            --batch_size 1000 \
            --neg_sample_size 50 \
            --init_size 0.001 \
            --learning_rate 0.05 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg \
            --multi_c 

# k=32, FB15K
python run.py \
            --dataset FB15K \
            --model ThreeH_TH \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adagrad \
            --max_epochs 500 \
            --patience 15 \
            --valid 5 \
            --batch_size 1000 \
            --neg_sample_size 200 \
            --init_size 0.001 \
            --learning_rate 0.2 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg \
            --multi_c 