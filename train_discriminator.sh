#!/usr/bin/env bash

THEANO_FLAGS='floatX=float32,device=gpu3' python main.py \
    --encoder 'lstm' \
    --batch_size 100 \
    --hidden_size 500 \
    --optimizer 'adam' \
    --lr 0.001 \
    --fine_tune_W True \
    --fine_tune_M True \
    --input_dir './twitter_dataset' \
    --dataset_fname 'dataset_HRED-VHRED-cTFIDF-rTFIDF-RND-TRUE_twitter_bpe.pkl' \
    --W_fname 'W_HRED-VHRED-cTFIDF-rTFIDF-RND-TRUE_300_twitter_bpe.pkl' \
    --n_epochs 100 \
    --patience 5 \
    --save_model True

