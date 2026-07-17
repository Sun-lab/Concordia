#!/bin/bash

# train the model on cords 2024 dataset using extended graph

data_name=cords_2024
data_dir=./data/Cords_data
graph_type=extended
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=48
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20
model_run_id=${data_name}__group_${group_scheme}__base16__rad48__topk${top_k}__ctg0p176__deg${degree_limit}

python  train.py \
    --data_name ${data_name} \
    --data_dir ${data_dir} \
    --graph_type ${graph_type} \
    --group_scheme ${group_scheme} \
    --dist_cutoff ${dist_cutoff} \
    --expanded_edge_cutoff ${expanded_edge_cutoff} \
    --top_k ${top_k} \
    --ctg_comp_dist_cutoff ${ctg_comp_dist_cutoff} \
    --degree_limit ${degree_limit} \
    --model_run_id ${model_run_id} \
    --device gpu \
    --s_dim2 40 \
    --batch_size 64
