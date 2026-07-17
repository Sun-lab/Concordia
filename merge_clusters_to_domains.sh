#!/bin/bash

mkdir -p merge_clusters_to_domains_Rout

data_name=cords_2024
data_dir=./data/Cords_data
graph_type=extended
n_kmeans_clusters=40
n_domains=10
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=48
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20
model_run_id=${data_name}__group_${group_scheme}__base16__rad48__topk${top_k}__ctg0p176__deg${degree_limit}

cur_name=${model_run_id}_${graph_type}_domains_k${n_domains}

R CMD BATCH --quiet --no-save \
"--args data_name='${data_name}' graph_type='${graph_type}' \
n_kmeans_clusters='${n_kmeans_clusters}' n_domains='${n_domains}' \
model_run_id='${model_run_id}' data_dir='${data_dir}'" \
merge_clusters_to_domains.R \
merge_clusters_to_domains_Rout/merge_clusters_to_domains_${cur_name}.Rout
