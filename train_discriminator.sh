#!/usr/bin/env bash

echo "VIRTUALENV = $VIRTUALENV"
. $VIRTUALENV/bin/activate

# nvidia-smi
# gpu-who

echo "THEANO_FLAGS = $THEANO_FLAGS"
python2.7 main.py \
    --data_path './twitter_dataset' \
    --dataset_fname 'dataset_RND-TRUE_twitter_bpe.pkl' \
    --W_fname 'W_300_twitter_bpe.pkl' \
    --save_prefix 'lstm-100_adam_RND-TRUE_twitter' \
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
