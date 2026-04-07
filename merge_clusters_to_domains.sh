#!/bin/bash

mkdir merge_clusters_to_domains_Rout

data_name=cords_2024
graph_type=extended
n_kmeans_clusters=40
n_domains=10

cur_name=${data_name}_${graph_type}

R CMD BATCH --quiet --no-save \
"--args data_name='${data_name}' graph_type='${graph_type}' \
n_kmeans_clusters='${n_kmeans_clusters}' n_domain='${n_domain}'" \
merge_clusters_to_domains.R \
merge_clusters_to_domains_Rout/merge_clusters_to_domains_${cur_name}.Rout

