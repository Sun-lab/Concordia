#!/bin/bash

# there are 1534 images in the processed Cords et al. 2024 dataset
# loop through all images
# this step can be run parallelly

for region_index in {0..1533}; do
    python  cluster_dist_in_image.py \
        --data_name cords_2024 \
        --graph_type extended \
        --region_index ${region_index} \
        --n_kmeans_clusters 40
done
