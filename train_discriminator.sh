#!/usr/bin/env bash

echo "VIRTUALENV = $VIRTUALENV"
. $VIRTUALENV/bin/activate

# nvidia-smi
# gpu-who

export THEANO_FLAGS=$THEANO_FLAGS,device=cuda
echo "THEANO_FLAGS = $THEANO_FLAGS"

CUDA_VERSION="8.0"
export CUDA=/usr/local/cuda-${CUDA_VERSION}/bin
export PATH=${CUDA}:$PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-${CUDA_VERSION}/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-${CUDA_VERSION}/lib64
export CPATH=$CPATH:/usr/local/cuda-${CUDA_VERSION}/lib64

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${gpuarray}/lib
export LIBRARY_PATH=$LIBRARY_PATH:${gpuarray}/lib
export CPATH=$CPATH:${gpuarray}/include

export HOME=/nlu/users/nicolas_gontier/home

echo "gpuarray = ${gpuarray}"
echo "CUDA = $CUDA"
echo "PATH = $PATH"
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "LIBRARY_PATH = $LIBRARY_PATH"
echo "CPATH = $CPATH"
echo "HOME = $HOME"

echo ""

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
