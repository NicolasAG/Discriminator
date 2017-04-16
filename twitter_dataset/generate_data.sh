#!/usr/bin/env bash

python combine_generated_data.py \
    --embedding_size 300\
    --data_fname_prefix dataset_HREDx2-VHREDx2-cTFIDF-RND-TRUE \
    --data_embeddings_prefix W_HREDx2-VHREDx2-cTFIDF-RND-TRUE \
    --inputs \
        ../../data/twitter/ModelResponses/HRED/HRED_20KVocab_BeamSearch_5_GeneratedTrainResponses_TopResponse.txt \
        ../../data/twitter/ModelResponses/HRED/HRED_Stochastic_GeneratedTrainResponses.txt \
        ../../data/twitter/ModelResponses/VHRED/VHRED_5000BPE_BeamSearch_5_GeneratedTrainResponses_TopResponse.txt \
        ../../data/twitter/ModelResponses/VHRED/VHRED_5000BPE_Stochastic_GeneratedTrainResponses.txt \
        ../../data/twitter/ModelResponses/TF-IDF/c_tfidf_TrainedResponses.txt \
    --random_model True \
    --oversampling True
