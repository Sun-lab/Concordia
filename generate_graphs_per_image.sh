#!/bin/bash

# there are 1534 images in the processed Cords et al. 2024 dataset
# loop through all images
# this step can be run parallelly

data_name=cords_2024
data_dir=./data/Cords_data
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=48
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20
model_run_id=${data_name}__group_${group_scheme}__base16__rad48__topk${top_k}__ctg0p176__deg${degree_limit}

for region_index in {0..1533}; do
    python generate_graphs_per_image.py \
          --data_name ${data_name} \
          --data_dir ${data_dir} \
          --region_index ${region_index} \
          --group_scheme ${group_scheme} \
          --dist_cutoff ${dist_cutoff} \
          --expanded_edge_cutoff ${expanded_edge_cutoff} \
          --top_k ${top_k} \
          --ctg_comp_dist_cutoff ${ctg_comp_dist_cutoff} \
          --degree_limit ${degree_limit} \
          --model_run_id ${model_run_id} \
          --prepare_folder graph_objects_prepare
done
