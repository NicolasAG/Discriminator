#!/usr/bin/env bash

THEANO_FLAGS='floatX=float32,device=gpu3' python main.py \
    --input_dir './twitter_dataset' \
    --dataset_fname 'dataset_HREDx2-VHREDx2-cTFIDF-RND-TRUE_twitter_bpe.pkl' \
    --W_fname 'W_HREDx2-VHREDx2-cTFIDF-RND-TRUE_300_twitter_bpe.pkl' \
    --save_model True \
    --save_prefix 'lstm-100_adam_HREDx2-VHREDx2-cTFIDF-RND-TRUE_twitter' \
    --batch_size 500 \
    --encoder 'lstm' \
    --hidden_size 100 \
    --is_bidirectional False \
    --n_recurrent_layers 1 \
    --patience 5 \
    --n_epochs 100 \
    --optimizer 'adam' \
    --fine_tune_W True \
    --fine_tune_M True
