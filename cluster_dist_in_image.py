#!/usr/bin/env python
# coding: utf-8

# within each image, compute the distance matrices among clusters
# one based on the embedding space distance and one based on the physical space distance


import numpy as np
import pandas as pd

from scipy.spatial import distance_matrix

from sklearn.cluster import KMeans
import datetime

import os
import sys
import argparse

from collections import defaultdict
from collections import Counter

from data_utilities import data_features


parser = argparse.ArgumentParser(description='compute embed and phyisical space distance')

parser.add_argument('--data_name', default="cords_2024", type=str, help='the name of the dataset to use')
parser.add_argument('--graph_type', default="extended", type=str,
                                    help='name for the feature and graph combination.',
                                    choices=['extended', 'basic', 'local'])
parser.add_argument('--n_kmeans_clusters', default=40, type=int, 
                                           help='the number of clusters to have from kmeans')     
parser.add_argument('--epoch_limit', default=1000, type=int, help='the number of epochs to train the model for, ' \
'which determines the epoch id of the embeddings to run k-means on')           

def get_dist_in_image(data_name="cords_2024", graph_type="extended", 
                      n_kmeans_clusters=40, epoch_limit=1000):

    input_args = locals()
    print("input args are", input_args)

    epoch_id = epoch_limit

    data_dicts = data_features(data_name, graph_type)

    train_images = data_dicts.train_images
    result_subfolder = data_dicts.result_subfolder
    n_cells_threshold = data_dicts.n_cells_threshold

    # Settings
    raw_dir = data_dicts.raw_dir  

    output_dir = "./results/"+result_subfolder+"/"+graph_type
    embedding_dir = output_dir+"/epoch_"+str(epoch_id)


    for image_ind in range(len(train_images)):

        cur_region = train_images[image_ind]

        df_cur = pd.read_csv(raw_dir+"/"+cur_region+".csv",
                            header=0)

        cur_coords = np.array([[x,y] for x,y in zip(df_cur["X"].tolist(),
                                                    df_cur["Y"].tolist())])
        cur_coords_mat = distance_matrix(cur_coords, cur_coords, p=2)

        # load the linear1 embedding values for the current region
        df_embedding = pd.read_csv(embedding_dir+"/linear1/linear1_"+cur_region+".csv",
                                header=0)

        df_embedding.shape

        assert df_cur.shape[0]==df_embedding.shape[0]

        np_embed = df_embedding.to_numpy()

        df_cluster = pd.read_csv(output_dir+"/kmeans_cluster.csv", 
                                header=0)
        
        # mapping from CELL_ID to kmeans cluster for all cells in the dataset
        cluster_dict = dict(zip(df_cluster["CELL_ID"].tolist(), 
                                df_cluster["kmeans_cluster"].tolist()))
        
        # get the kmeans cluster for each cell in the current region
        cur_kmeans_clusters = [cluster_dict[x] for x in df_cur["CELL_ID"].tolist()]

        embed_mat = np.full((n_kmeans_clusters, n_kmeans_clusters), np.nan)
        coord_mat = np.full((n_kmeans_clusters, n_kmeans_clusters), np.nan)

        counter_cluster = Counter(cur_kmeans_clusters)

        for i in range(n_kmeans_clusters-1):
            if counter_cluster[i] >= n_cells_threshold:
                rows_i = [x for x,y in enumerate(cur_kmeans_clusters) if y==i]
                embed_slice_i = np_embed[rows_i]
                embed_i = np.mean(embed_slice_i, axis=0)
                for j in range(i+1, n_kmeans_clusters):
                    if counter_cluster[j] >= n_cells_threshold:
                        rows_j = [x for x,y in enumerate(cur_kmeans_clusters) if y==j]
                        embed_slice_j = np_embed[rows_j]
                        embed_j = np.mean(embed_slice_j, axis=0)
                        embed_l2 = np.linalg.norm(embed_i - embed_j)
                        coords_mat_sub = cur_coords_mat[rows_i][:, rows_j]
                        min_i = np.min(coords_mat_sub, axis=1)
                        min_j = np.min(coords_mat_sub, axis=0)
                        q_i = np.quantile(min_i, 0.9)
                        q_j = np.quantile(min_j, 0.9)
                        min_q = min([q_i, q_j])
                        embed_mat[i,j] = embed_l2
                        coord_mat[i,j] = min_q
                        embed_mat[j,i] = embed_l2
                        coord_mat[j,i] = min_q

        embed_output_dir = os.path.join(output_dir, "embedding_dist_in_image")
        coord_output_dir = os.path.join(output_dir, "coord_dist_in_image")
        os.makedirs(embed_output_dir, exist_ok=True)
        os.makedirs(coord_output_dir, exist_ok=True)

        np.savetxt(os.path.join(embed_output_dir, f"embedding_dist_{cur_region}.csv"), 
                embed_mat, delimiter=",")
                
        np.savetxt(os.path.join(coord_output_dir, f"coord_dist_{cur_region}.csv"), 
                coord_mat, delimiter=",")
        
        if image_ind % 100 == 0:
            print("done with image index "+str(image_ind))


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    get_dist_in_image(**vars(args))
