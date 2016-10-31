#!/usr/bin/env bash

THEANO_FLAGS='floatX=float32,device=gpu0' python main.py \
    --encoder 'lstm' \
    --batch_size 256 \
    --hidden_size 300 \
    --optimizer 'adam' \
    --lr 0.001 \
    --fine_tune_W True \
    --fine_tune_M True \
    --input_dir './twitter_dataset' \
    --dataset_fname 'dataset_twitter_bpe_indices.pkl' \
    --W_fname 'W_twitter_bpe_rand.pkl' \
    --n_epochs 10 \
    --save_model True #\
    # --train_examples 1000
