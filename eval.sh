#!/usr/bin/env bash

THEANO_FLAGS='base_compiledir=/work/.theano-$USER,floatX=float32,device=cuda' pytho2.7 main.py \
    --data_path './twitter_dataset' \
    --dataset_fname 'dataset_RND-TRUE_twitter_bpe.pkl' \
    --W_fname 'W_300_twitter_bpe.pkl' \
    --load_path './trained_models/lstm-100_RND' \
    --load_prefix 'lstm-100_adam_RND-TRUE_twitter' \
    --test True \
    --plot_response_length True \
    --plot_learning_curves True
