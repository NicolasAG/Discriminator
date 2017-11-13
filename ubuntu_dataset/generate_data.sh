#!/usr/bin/env bash

python combine_generated_data_words.py \
    --embedding_size 300 \
    --data_embeddings_prefix W \
    --data_fname_prefix dataset_RND-TRUE \
    --random_model True
