#!/usr/bin/env bash

echo "VIRTUALENV = $VIRTUALENV"
. $VIRTUALENV/bin/activate

export THEANO_FLAGS=$THEANO_FLAGS,device=gpu
echo "THEANO_FLAGS = $THEANO_FLAGS"

CUDA_VERSION="8.0"
export CUDA=/usr/local/cuda-${CUDA_VERSION}/bin
export PATH=${CUDA}:$PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-${CUDA_VERSION}/lib64
# export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-${CUDA_VERSION}/lib64
# export CPATH=$CPATH:/usr/local/cuda-${CUDA_VERSION}/lib64

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${gpuarray}/lib
# export LIBRARY_PATH=$LIBRARY_PATH:${gpuarray}/lib
# export CPATH=$CPATH:${gpuarray}/include

# To use P100 GPUs:
export CUDA_CACHE_PATH=/lm/scratch/nicolas_gontier/cuda_cache

export HOME=/nlu/users/nicolas_gontier/home

echo "gpuarray = ${gpuarray}"
echo "CUDA = $CUDA"
echo "PATH = $PATH"
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "LIBRARY_PATH = $LIBRARY_PATH"
echo "CPATH = $CPATH"
echo "HOME = $HOME"

echo ""

# python2.7 main.py \
#     --data_path './twitter_dataset' \
#     --dataset_fname 'dataset_RND-TRUE_twitter_bpe.pkl' \
#     --W_fname 'W_300_twitter_bpe.pkl' \
#     --load_path './trained_models/twitter_lstm-100' \
#     --load_prefix 'lstm-100_adam_twitter' \
#     --save_path '.' \
#     --save_prefix 'lstm-100_adam_twitter' \
#     --retrieve True

python2.7 main.py \
    --data_path './ubuntu_dataset/v2' \
    --dataset_fname 'dataset.pkl' \
    --W_fname 'W.pkl' \
    --load_path './trained_models/ubuntu-v2_lstm-200' \
    --load_prefix 'lstm-200_adam_ubuntu-v2' \
    --save_path '.' \
    --save_prefix 'lstm-200_adam_ubuntu-v2' \
    --test True \
    --plot_learning_curves False
