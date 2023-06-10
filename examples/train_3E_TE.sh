#!/bin/bash

# k=32, WN18RR
python run.py \
            --dataset WN18RR \
            --model ThreeE_TE \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adagrad \
            --max_epochs 400 \
            --patience 15 \
            --valid 5 \
            --batch_size 500 \
            --neg_sample_size 100 \
            --init_size 0.001 \
            --learning_rate 0.2 \
            --gamma 0.0 \
            --bias learn \
            --dtype single \
            --double_neg 

# k=32, FB237
python run.py \
            --dataset FB237 \
            --model ThreeE_TE \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adagrad \
            --max_epochs 400 \
            --patience 15 \
            --valid 5 \
            --batch_size 1000 \
            --neg_sample_size 50 \
            --init_size 0.001 \
            --learning_rate 0.05 \
            --gamma 0.0 \
            --bias learn \
            --dtype single \
            --double_neg 

# k=32, FB15K
python run.py \
            --dataset FB15K \
            --model ThreeE_TE \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adagrad \
            --max_epochs 400 \
            --patience 15 \
            --valid 5 \
            --batch_size 1000 \
            --neg_sample_size 200 \
            --init_size 0.001 \
            --learning_rate 0.2 \
            --gamma 0.0 \
            --bias learn \
            --dtype single \
            --double_neg 