#!/bin/bash

# Train the default setting 10 times with train.py, once per triple of random
# seeds in data/random_seeds_10_triples.txt (30 seeds = 10 disjoint triples).
#
# model_run_id is passed unsuffixed and it is what locates the graph objects to load. 
# train_rs.py appends
# __repeat_<random_seed_index> itself when building its output paths, so the runs
# land in separate folders:
#   results_rs/<model_run_id>__repeat_<i>/<graph_type>
#   saved_models_rs/<model_run_id>__repeat_<i>/<graph_type>


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
group_scheme=default
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20
lr=0.0001
epoch_limit=1000
random_seed_file=./data/random_seeds_10_triples.txt

# on large datasets, each training run can take long
# and it is recommended to submitt the computing jobs to run in parallel

for i in {1..10}; do
for dist_cutoff in 16; do
    expanded_edge_cutoff=$((3 * dist_cutoff))
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}

    python train.py \
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
            --random_seed_index ${i} \
            --random_seed_file ${random_seed_file} \
            --device gpu \
            --s_dim2 40 \
            --batch_size 64 \
            --lr ${lr} \
            --epoch_limit ${epoch_limit}"
done
done
