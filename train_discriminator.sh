#!/usr/bin/env bash

THEANO_FLAGS='floatX=float32,device=gpu3' python main.py \
    --input_dir './twitter_dataset' \
    --dataset_fname 'dataset_HRED-VHRED-cTFIDF-rTFIDF-RND-TRUE_twitter_bpe.pkl' \
    --W_fname 'W_HRED-VHRED-cTFIDF-rTFIDF-RND-TRUE_300_twitter_bpe.pkl' \
    --save_model True \
    --save_prefix 'bi-lstm-50_adam_HRED-VHRED-cTFIDF-rTFIDF-RND-TRUE_twitter' \
    --batch_size 500 \
    --encoder 'lstm' \
    --hidden_size 50 \
    --is_bidirectional True \
    --n_recurrent_layers 1 \
    --patience 5 \
    --n_epochs 100 \
    --optimizer 'adam' \
    --fine_tune_W True \
    --fine_tune_M True
