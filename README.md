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