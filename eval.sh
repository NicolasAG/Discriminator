#!/usr/bin/env bash

# echo "VIRTUALENV = $VIRTUALENV"
# . $VIRTUALENV/bin/activate

# export THEANO_FLAGS=$THEANO_FLAGS,device=gpu
# echo "THEANO_FLAGS = $THEANO_FLAGS"

# CUDA_VERSION="8.0"
# export CUDA=/usr/local/cuda-${CUDA_VERSION}/bin
# export PATH=${CUDA}:$PATH

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-${CUDA_VERSION}/lib64
# # export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-${CUDA_VERSION}/lib64
# # export CPATH=$CPATH:/usr/local/cuda-${CUDA_VERSION}/lib64

# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${gpuarray}/lib
# # export LIBRARY_PATH=$LIBRARY_PATH:${gpuarray}/lib
# # export CPATH=$CPATH:${gpuarray}/include

# # To use P100 GPUs:
# export CUDA_CACHE_PATH=/lm/scratch/nicolas_gontier/cuda_cache

# export HOME=/nlu/users/nicolas_gontier/home

# echo "gpuarray = ${gpuarray}"
# echo "CUDA = $CUDA"
# echo "PATH = $PATH"
# echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
# echo "LIBRARY_PATH = $LIBRARY_PATH"
# echo "CPATH = $CPATH"
# echo "HOME = $HOME"

# echo ""

#THEANO_FLAGS='floatX=float32,device=cpu,force_device=True' python main.py \
THEANO_FLAGS='floatX=float32,device=gpu2' python main.py \
    --data_path '../data/convai/de' \
    --dataset_fname 'round1_DE.dataset.pkl' \
    --dict_fname 'round1_DE.dict.pkl' \
    --load_path './trained_models/convai-h2h_exp4' \
    --load_prefix 'convai-h2h_exp4' \
    --save_path '.' \
    --save_prefix 'convai-h2h_exp4' \
    --test True \
    --plot_learning_curves False

