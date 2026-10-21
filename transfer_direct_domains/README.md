# Transfer domains from a model trained on a different dataset

Example transfers the domains learnt from the Cords et al. 2024 non small-cell lung cancer dataset to an independent lung cancer dataset from Lermi et al. 2025.

These code should be put and run at the same folder level as those outside this folder. 

### Prerequisites

The transfer applies when

- the two datasets are from the same type of tissues
- the fine-scale cell types of the independent dataset can be grouped into the same set of coarse-scale cell types as used by the training dataset
- the units of physical distance in the two datasets are the same
- the domains of the training dataset were obtained directly through running K-means on the embeddings, following [train_direct_domains](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains)

Before the transfer, finish all the steps in [train_direct_domains](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains) on the training dataset. The transfer uses the 10 trained models saved by `train_rs_10.sh`, the cell embeddings of the training dataset from these models, and the `ensemble_runs.csv` and `domain_annotation_for_cells.csv` files saved by `k_means_on_embeddings_direct_domains.sh`.

### Prepare to generate graphs for the independent dataset

Add a configuration block for the independent dataset in the `data_features` class in [data_utilities.py](https://github.com/Sun-lab/Concordia/blob/main/data_utilities.py), as described in Item 2 of the [main page](https://github.com/Sun-lab/Concordia). The `group_ct_mapping` must use the same coarse-scale cell type names as the training dataset. 

The graph parameters (`dist_cutoff`, `expanded_edge_cutoff`, `top_k`, `ctg_comp_dist_cutoff`, `degree_limit`) should be the same as those used for the training dataset.

### Generate graph objects for each image one by one

for computational time consideration, it is recommended to generate graph objects for each image in a dataset separately and run in parallel

[generate_graphs_per_image_lermi.sh](https://github.com/Sun-lab/Concordia/blob/main/transfer_direct_domains/generate_graphs_per_image_lermi.sh)

which calls the file 

[generate_graphs_per_image.py](https://github.com/Sun-lab/Concordia/blob/main/generate_graphs_per_image.py)

### Reorganize the graph objects and other relevant files:

[reorganize_graph_objects_lermi.py](https://github.com/Sun-lab/Concordia/blob/main/transfer_direct_domains/reorganize_graph_objects_lermi.py)


### Transfer the domains to cells in the independent dataset

[transfer_direct_domains_lermi.sh](https://github.com/Sun-lab/Concordia/blob/main/transfer_direct_domains/transfer_direct_domains_lermi.sh)

which calls

[transfer_direct_domains.py](https://github.com/Sun-lab/Concordia/blob/main/transfer_direct_domains/transfer_direct_domains.py)

No model is trained in this step. Each of the 10 trained models is applied to the graphs of the independent dataset to save out cell embeddings. The embeddings of both datasets are scaled with the per-model scale factors of the training dataset and concatenated across the 10 models. Each cell in the independent dataset is then assigned to the domain whose centroid, computed from the training dataset cells, is the nearest.


The final output file is `domain_annotation_for_cells.csv`, saved under

```
results_rs_transfer/<target_model_run_id>/<source_model_run_id>__ensemble_10/extended/domains_k10
```

which gives the correspondance between `CELL_ID` and `domain` annotation for all cells from all images in the independent dataset. The domain labels are the same as those of the training dataset.
