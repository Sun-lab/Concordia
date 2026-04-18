#!/bin/bash

# there are 1534 images in the processed Cords et al. 2024 dataset
# loop through all images
# this step can be run parallelly

for region_index in {0..1533}; do
    python generate_graphs_per_image.py \
          --data_name cords_2024 \
          --data_dir ./data/Cords_data \
          --region_index ${region_index} \
          --degree_limit 20 \
          --prepare_folder graph_objects_prepare
done
