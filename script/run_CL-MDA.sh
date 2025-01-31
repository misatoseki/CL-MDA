#!/bin/bash

cd /path/to/CL-MDA

DATETIME=$(date '+%Y%m%d_%H%M%S')
LOGPATH="../logs_CL-MDA/experiment_${DATETIME}_${RANDOM}"
mkdir -p ${LOGPATH}

LOG="${LOGPATH}/CL-MDA.log"

python3 main.py --epoch 1500 \
                --batch_size 16 \
                --margin 1 \
                --lr 1e-5 \
                --hidden_dim1 512 \
                --hidden_dim2 256 \
                --emb_dim 64 \
                --pos_weight 10 \
                --neg_weight 1 \
                --path_microbe_emb "data/dataset_MDAD_microbe_evo.pk" \
                --path_drug_emb "data/dataset_MDAD_drug_molformer.pk" \
                --logpath ${LOGPATH} \
                --path_adj "../indata/MDAD/adj_new.txt" | tee $LOG


