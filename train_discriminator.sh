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

# python2.7 compute_response_encoddings.py -v trained_models/ubuntu-bpe-5k_lstm-200/lstm-200_adam_ubuntu-bpe-5k

python2.7 main.py \
    --data_path './ubuntu_dataset/bpe_v2' \
    --dataset_fname 'DE_BPE-5k-dataset.pkl' \
    --W_fname 'W_BPE-5k-rnd.pkl' \
    --save_path '.' \
    --save_prefix 'bi-lstm-200_adam_ubuntu-bpe-5k' \
    --batch_size 256 \
    --encoder 'lstm' \
    --hidden_size 200 \
    --is_bidirectional True \
    --dropout_p 0. \
    --n_recurrent_layers 1 \
    --patience 10 \
    --n_epochs 100 \
    --optimizer 'adam' \
    --seed 42 \
    --fine_tune_W True \
    --fine_tune_M True #\
    #--load_path './trained_models/ubuntu-v2_lstm-200' \
    #--load_prefix 'lstm-200_adam_ubuntu-v2' \
    #--resume True
