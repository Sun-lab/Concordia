# Concordia
Spatial Domain Detection via Augmented Graphs for Population-Level Spatial Proteomics

## Step 1. Prepare raw data files

Raw data files involve two types of files:

1. Dataset-level file: `region_list.csv`, a .csv file listing the names of all images/tissues in the dataset. The format requirement on this .csv file is:
   - the required column is `region_ID`, with each row corresponding to one unique `region_ID` name. 

2. Image/tissue-level files: one .csv file for each image/tissue, which each row representing one cell from the corresponding image. All these .csv files from the same dataset are put under one folder. The format requirements on these files are:
   - the name of the .csv file for each image has the format `${region_ID}.csv`. 
   - the required columns are `CELL_ID`, `X`, `Y`, `CELL_TYPE`
     - `CELL_ID`: an integer index for a cell that is **unique among all cells in the entire dataset** (unique across all images)
     - `X`, `Y`: numerical values, x and y coordinates for the spatial location of a cell
     - `CELL_TYPE`: strings, fine scale cell types, can be mapped to coarser scale cell types for use in the model

Processed files for the lung cancer dataset (originally version of data is from [Cords et al. 2024](https://www.cell.com/cancer-cell/fulltext/S1535-6108(23)00449-X)) used in the paper are included under the folder
```
data/Cords_data
```

## Step 2. Generate graphs

To generate graphs for a new dataset, add a new named configuration block in the `data_features` class in `data_utilities.py` by adding an `if data_name == "<your_dataset_name>":` block and specifying the following fields:

1. Paths:
   - `raw_dir`: path to the folder containing the per-image/tissue .csv files prepared in Step 1
   - `dataset_root`: path to the folder where the generated graph objects will be saved

2. Cell type mappings:
   - `cell_type_mapping`: a dictionary mapping each fine-scale cell type name (string) to a **unique integer between 0 and number of cell types - 1**
   - `group_ct_mapping`: a dictionary mapping each coarse-scale cell type name (string) to a set of fine-scale cell type names that belong to that group

3. Region list:
   - `train_images`: a list of `region_ID` strings specifying which images/tissues to include; typically loaded from `region_list.csv` prepared in Step 1

4. Other parameters:
   - `dist_cutoff`: numerical distance cutoff for adding edges in the basic graph, changes according to different datasets (e.g., `16`)
   - `path_purity_cutoff`: a value between 0 and 1; the edges between pairs of cells having shortest paths with purity below this threshold are excluded as candidate edges in the second step extension, default `0.90`
   - `n_cells_threshold`: integer minimum number of cells a cluster must contain in a given image to be included when computing embedding and physical distances between clusters per image, changes according to different datasets (e.g., `30`)

For computational time consideration, it is recommended to generate graph objects for each image in a dataset separately and run in parallel, and move individual files together to one folder. 

The step of generating graph objects for each image one by one (can be parallelized), taking Cords et al. 2024 data as an example, can be done by running 

[generate_graphs_per_image.sh](https://github.com/Sun-lab/Concordia/blob/main/generate_graphs_per_image.sh)

which calls the file 

[generate_graphs_per_image.py](https://github.com/Sun-lab/Concordia/blob/main/generate_graphs_per_image.py)

The moving of the graph objects and other relevant files can be done by:

[reorganize_graph_objects.ipynb](https://github.com/Sun-lab/Concordia/blob/main/reorganize_graph_objects.ipynb)

Notes on random seed setting:

Random seeds can be set for the graph generation process by modifying the code. Although, even under the contol of a fixed random seed for each image, the agumented graphs for the same image from two runs may have difference, due to the ranking difference caused by floating number accuracy issue in the step 1 graph extension step (local graph). 

## Step3. Train GNN model with sparse_mincut_pool loss and save cell embeddings out

Taking Cords et al. 2024 data as an example, the model training and embedding saving can be done by

[train.sh](https://github.com/Sun-lab/Concordia/blob/main/train.sh)

which calls

[train.py](https://github.com/Sun-lab/Concordia/blob/main/train.py)

The `sparse_mincut_pool` function called has a documentation [here](https://github.com/Sun-lab/sparse_mincut_pool/tree/main).

## Step4. Obtain domains

### step4.1 Run K-means on embeddings to get a relatively large number of clusters

Taking Cords et al. 2024 data as an example, this step can be done by

[k_means_on_embeddings.sh](https://github.com/Sun-lab/Concordia/blob/main/k_means_on_embeddings.sh)

which calls

[k_means_on_embeddings.py](https://github.com/Sun-lab/Concordia/blob/main/k_means_on_embeddings.py)

### step4.2 Within each image, compute the embedding distance and physical distance between any pair of clusters

Taking Cords et al. 2024 data as an example, this step (can be parallelized) can be done by

[cluster_dist_in_image.sh](https://github.com/Sun-lab/Concordia/blob/main/cluster_dist_in_image.sh)

which calls

[cluster_dist_in_image.py](https://github.com/Sun-lab/Concordia/blob/main/cluster_dist_in_image.py)


### step4.3 Get final weighted distance matrix, merge clusters into domains

To get the final weighted distance matrix, first specify the parameters according to the dataset in the file `data_utilities.R`:

1. R data frame:
   - `df`: the R data frame of the image information. typically loaded from `region_list.csv` prepared in Step 1

2. Paths:
   - `raw_dir`: path to the folder containing the per-image/tissue .csv files prepared in Step 1

3. Other parameters:
   - `n_cells_threshold`: integer minimum number of cells a cluster must contain in a given image to be included when computing embedding and physical distances between clusters per image, changes according to different datasets (e.g., `30`)
   - `n_images_threshold`: the number of images threshold for deciding for each pair of clusters, whether to aggregate the distances across images. Only aggregate if there are at least `n_images_threshold` images with at least `n_cells_threshold` in each of the two clusters. Changes according to different datasets (e.g., `30`).

After preparing the parameters, taking Cords et al. 2024 data as an example, this step of getting final domain annotation can be done by

[merge_clusters_to_domains.sh](https://github.com/Sun-lab/Concordia/blob/main/merge_clusters_to_domains.sh)

which calls

[merge_clusters_to_domains.R](https://github.com/Sun-lab/Concordia/blob/main/merge_clusters_to_domains.R)

The final output file is `domain_annotation_for_cells.csv`, which gives the correspondance between `CELL_ID` and `domain` annotation for all cells from all images in the dataset. 
