![Python 3.8.18](https://shields.io)
![R 4.4.0](https://shields.io)

# Concordia
Spatial Domain Detection via Augmented Graphs for Population-Level Spatial Proteomics

This repo is organized such that the code files in the current layer are needed for all settings, while those specific to each settings are under each folder. 

## Preparation

### Item 1. Prepare raw data files

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

3. Random seed file for model training (required): a file with random seeds for training an ensemble of 10 GNN models. Each run needs 3 random seeds, for torch.manual_seed, random.seed, and np.random.seed. So 30 random seeds are needed. The default example random seed file used for model training is [random_seeds_10_triples.txt](https://github.com/Sun-lab/Concordia/tree/main/data/random_seeds_10_triples.txt). 

4. Random seed file for generating graphs (optional): a file with random seeds for generating the graphs for each image, with the number of random seeds equal to the number of images. If not provided, random seed 0 will be used for all images. An example random seed file for the lung cancer dataset is [random_seeds.txt](https://github.com/Sun-lab/Concordia/tree/main/data/Cords_data). 

### Item 2. Prepare to generate graphs

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

## Obtain the domains

There are three ways to obtain the domains. For each approach, code files for one example dataset organized under a corresponding folder, with a Readme file. 


### Train model and directly obtain domains from embeddings

Recommended for images with complex spatial structures, for example, tumor invasive front:

[train_direct_domains](https://github.com/Sun-lab/Concordia/blob/main/train_direct_domains)

### Train model and directly obtain domains in a two-step procedure

Recommended for well structured tissues, for example, mouse medial prefrontal cortex,:

[train_two_steps](https://github.com/Sun-lab/Concordia/blob/main/train_two_steps)

### Transfer domains from a model trained on a different dataset

Given the models and domains obtained from a model trained on a training dataset, for independent dataset, if it is from the same type of tissues, the fine-scale cell types can be grouped into the same set of coarse-scale cell type as used by the training data, and the units of physical distance in two datasets are the same, then domains learnt from the training dataset can be transferred to cells in the independent dataset. This applies when the domains learnt from the training dataset were obtained directly through running K-means on the embeddings. 


[transfer_direct_domains](https://github.com/Sun-lab/Concordia/blob/main/transfer_direct_domains)



## Trained models, raw embeddings, scaling factors and domain annotations 

These materials for the default Concordia trained on main non small-cell lung cancer dataset (Cords. et al. 2024) are available through this link on Google Drive:

[Concordia_materials](https://drive.google.com/file/d/1XyqGVWV4iM5LeeN5H-AkTyt0XQEZ_iuD/view?usp=sharing)

From each of the 10 trained GNN models, one embedding file was saved out for each image. The embedding file does not keep the cell ID, but it has row correspondence with the cells in the raw data file as in 
```
data/Cords_data/raw_data.tar.gz
```
which has the cell ID.

