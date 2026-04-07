#!/bin/bash

data_name="cords_2024"
graph_type="extended"
n_kmeans_clusters=40


python k_means_on_embeddings.py \
    --data_name ${data_name} \
    --graph_type ${graph_type} \
    --n_kmeans_clusters ${n_kmeans_clusters}
