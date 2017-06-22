#!/usr/bin/env bash

echo "VIRTUALENV = $VIRTUALENV"
. $VIRTUALENV/bin/activate

export THEANO_FLAGS=$THEANO_FLAGS,device=gpu,optimizer=fast_compile  # ,exception_verbosity=high
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

python2.7 compute_response_encoddings.py -v trained_models/ubuntu-bpe-5k_lstm-200/lstm-200_adam_ubuntu-bpe-5k

