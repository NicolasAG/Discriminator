#!/usr/bin/env bash

THEANO_FLAGS='floatX=float32,device=gpu0' python main.py \
    --data_path './twitter_dataset' \
    --dataset_fname 'dataset_HREDx2-VHREDx2-cTFIDF-RND-TRUE_twitter_bpe.pkl' \
    --W_fname 'W_HREDx2-VHREDx2-cTFIDF-RND-TRUE_300_twitter_bpe.pkl' \
    --load_prefix 'lstm-100_adam_HREDx2-VHREDx2-cTFIDF-RND-TRUE_twitter' \
    --test True \
    --plot_human_scores True \
    --plot_response_length True \
    --plot_learning_curves True
