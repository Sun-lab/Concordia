# Obtain domains from the two-step procedure

Example uses mouse brain data.



### Generate graph objects for each image one by one

for computational time consideration, it is recommended to generate graph objects for each image in a dataset separately and run in parallel

[generate_graphs_per_image.sh](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains/generate_graphs_per_image.sh)

which calls the file 

[generate_graphs_per_image.py](https://github.com/Sun-lab/Concordia/blob/main/generate_graphs_per_image.py)

### Reorganize the graph objects and other relevant files:

[reorganize_graph_objects.py](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains/reorganize_graph_objects.py)


### Train GNN model with sparse_mincut_pool loss and save cell embeddings out


[train_rs_10.sh](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains/train_rs_10.sh)

which calls

[train_rs.py](https://github.com/Sun-lab/Concordia/blob/main/train_rs.py)

The `sparse_mincut_pool` function called has a documentation [here](https://github.com/Sun-lab/sparse_mincut_pool/tree/main).

### Run K-means on embeddings to get a relatively large number of clusters

[k_means_on_embeddings_direct_domains.sh](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains/k_means_on_embeddings_direct_domains.sh)

which calls

[k_means_on_embeddings_direct_domains.py](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains/k_means_on_embeddings_direct_domains.py)



### Within each image, compute the embedding distance and physical distance between any pair of clusters


[cluster_dist_in_image_scaled_ensemble_mpfc.sh](https://github.com/Sun-lab/Concordia/blob/main/train_two_steps/cluster_dist_in_image_scaled_ensemble_mpfc.sh)

which calls

[cluster_dist_in_image_scaled_ensemble.py](https://github.com/Sun-lab/Concordia/blob/main/train_two_steps/cluster_dist_in_image_scaled_ensemble.py)


### Get final weighted distance matrix between clusters, and merge clusters to domains

To get the final weighted distance matrix, first specify the parameters according to the dataset in the file `data_utilities.R`:

1. R data frame:
   - `df`: the R data frame of the image information. typically loaded from `region_list.csv` prepared in Step 1

2. Paths:
   - `raw_dir`: path to the folder containing the per-image/tissue .csv files prepared in Step 1

3. Other parameters:
   - `n_cells_threshold`: integer minimum number of cells a cluster must contain in a given image to be included when computing embedding and physical distances between clusters per image, changes according to different datasets (e.g., `30`)
   - `n_images_threshold`: the number of images threshold for deciding for each pair of clusters, whether to aggregate the distances across images. Only aggregate if there are at least `n_images_threshold` images with at least `n_cells_threshold` in each of the two clusters. Changes according to different datasets (e.g., `30`).


[merge_clusters_to_domains.sh](https://github.com/Sun-lab/Concordia/blob/main/merge_clusters_to_domains.sh)

which calls

[merge_clusters_to_domains.R](https://github.com/Sun-lab/Concordia/blob/main/merge_clusters_to_domains.R)

The final output file is `domain_annotation_for_cells.csv`, which gives the correspondance between `CELL_ID` and `domain` annotation for all cells from all images in the dataset.

