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

python2.7 main.py \
    --data_path './ubuntu_dataset' \
    --dataset_fname 'dataset_RND-TRUE_ubuntu_bpe.pkl' \
    --W_fname 'W_300_ubuntu_bpe.pkl' \
    --save_path '.' \
    --save_prefix 'lstm-300_adam_RND-TRUE_ubuntu' \
    --batch_size 256 \
    --encoder 'lstm' \
    --hidden_size 300 \
    --is_bidirectional False \
    --n_recurrent_layers 1 \
    --patience 10 \
    --n_epochs 100 \
    --optimizer 'adam' \
    --fine_tune_W True \
    --fine_tune_M True \
    --load_path './trained_models/lstm-300_RND' \
    --load_prefix 'lstm-300_adam_RND-TRUE_ubuntu' #\
    #--resume True
