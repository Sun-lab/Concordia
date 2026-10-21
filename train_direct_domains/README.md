# Directly obtain the domains from K-means on final embeddings

Example use Cords et al. 2024 non small-cell lung cancer dataset.

These code should be put and run at the same folder level as those outside this folder. 

### Generate graph objects for each image one by one

for computational time consideration, it is recommended to generate graph objects for each image in a dataset separately and run in parallel

[generate_graphs_per_image.sh](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains/generate_graphs_per_image.sh)

which calls the file 

[generate_graphs_per_image.py](https://github.com/Sun-lab/Concordia/blob/main/generate_graphs_per_image.py)

### Reorganize the graph objects and other relevant files:

[reorganize_graph_objects.py](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains/reorganize_graph_objects.py)


### Train GNN model with sparse_mincut_pool loss and save cell embeddings out

Taking Cords et al. 2024 data as an example, the model training and embedding saving can be done by

[train_rs_10.sh](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains/train_rs_10.sh)

which calls

[train_rs.py](https://github.com/Sun-lab/Concordia/blob/main/train_rs.py)

The `sparse_mincut_pool` function called has a documentation [here](https://github.com/Sun-lab/sparse_mincut_pool/tree/main).

### Run K-means on embeddings to directly get final domains


[k_means_on_embeddings_direct_domains.sh](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains/k_means_on_embeddings_direct_domains.sh)

which calls

[k_means_on_embeddings_direct_domains.py](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains/k_means_on_embeddings_direct_domains.py)