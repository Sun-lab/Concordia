# Concordia
Consistent spatial domain detection across tissues via graph neural networks on augmented graphs

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

The step of generating graph objects for each image one by one, taking Cords et al. 2024 data as an example, can be done by running 

[generate_graphs_per_image.sh](https://github.com/Sun-lab/Concordia/blob/main/generate_graphs_per_image.sh)

which calls the file 

[generate_graphs_per_image.py](https://github.com/Sun-lab/Concordia/blob/main/generate_graphs_per_image.py)

The moving of the graph objects and other relevant files can be done by:

[reorganize_graph_objects.ipynb](https://github.com/Sun-lab/Concordia/blob/main/reorganize_graph_objects.ipynb)

## Step3. Train GNN model with sparse_mincut_pool loss

Taking Cords et al. 2024 data as an example, the model training can be done by

[train.sh](https://github.com/Sun-lab/Concordia/blob/main/train.sh)

which calls

[train.py](https://github.com/Sun-lab/Concordia/blob/main/train.py)