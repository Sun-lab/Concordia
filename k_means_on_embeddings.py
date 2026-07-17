#!/usr/bin/env python
# coding: utf-8

# run k-means on the embedding space of the trained GNN model


import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from datetime import datetime

import os
import sys
import argparse

from collections import defaultdict
from collections import Counter

from data_utilities import data_features, make_model_run_id


parser = argparse.ArgumentParser(description='k-means on model embedding space')

parser.add_argument('--data_name', default="cords_2024", type=str, help='the name of the dataset to use')
parser.add_argument('--graph_type', default="extended", type=str,
                                    help='name for the feature and graph combination.',
                                    choices=['extended', 'basic', 'local'])                               
parser.add_argument('--n_kmeans_clusters', default=40, type=int, help='the number of clusters to have from kmeans')
parser.add_argument('--epoch_limit', default=1000, type=int, help='the number of epochs to train the model for, ' \
'which determines the epoch id of the embeddings to run k-means on')
parser.add_argument('--data_dir', default="./data/Cords_data", type=str, help='the overall folder of the dataset')
parser.add_argument('--dist_cutoff', type=float, default=16, help='distance cutoff for constructing the base graph')
parser.add_argument('--expanded_edge_cutoff', type=float, default=48, help='search radius for candidate cells in the first extension step')
parser.add_argument('--top_k', type=int, default=4, help='number of nearest candidate cells to add in the first extension step')
parser.add_argument('--ctg_comp_dist_cutoff', type=float, default=0.176, help='cell type group composition distance cutoff for adding edges')
parser.add_argument('--degree_limit', type=int, default=20, help='desired average degree of nodes in each fully extended graph')
parser.add_argument('--group_scheme', type=str, default="default", help='fine-to-coarse cell type grouping scheme')
parser.add_argument('--model_run_id', type=str, default=None, help='dataset-aware identifier for this graph/model parameter set')

def run_k_means(data_name="cords_2024",
                graph_type="extended",
                n_kmeans_clusters=40,
                epoch_limit=1000,
                data_dir="./data/Cords_data",
                dist_cutoff=16,
                expanded_edge_cutoff=48,
                top_k=4,
                ctg_comp_dist_cutoff=0.176,
                degree_limit=20,
                group_scheme="default",
                model_run_id=None):

    input_args = locals()
    print("input args are", input_args)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start running K-means on embeddings: ", current_time)

    n_kmeans_clusters = int(n_kmeans_clusters)

    epoch_id = epoch_limit

    if model_run_id is None:
        model_run_id = make_model_run_id(data_name, group_scheme, dist_cutoff,
                                         expanded_edge_cutoff, top_k,
                                         ctg_comp_dist_cutoff, degree_limit)

    data_dicts = data_features(data_name, graph_type,
                               group_scheme=group_scheme,
                               dist_cutoff=dist_cutoff,
                               expanded_edge_cutoff=expanded_edge_cutoff,
                               top_k=top_k,
                               ctg_comp_dist_cutoff=ctg_comp_dist_cutoff,
                               degree_limit=degree_limit,
                               model_run_id=model_run_id,
                               data_dir=data_dir)

    train_images = data_dicts.train_images

    result_subfolder = data_dicts.result_subfolder

    # Settings
    raw_dir = data_dicts.raw_dir

    output_dir = "./results/"+result_subfolder+"/"+graph_type
    embedding_dir = output_dir+"/epoch_"+str(epoch_id)

    layer_chunk = []
    num_id_list = []

    for i in range(len(train_images)):

        cur_region = train_images[i]

        df_cur = pd.read_csv(raw_dir+"/"+cur_region+".csv",
                            header=0)

        # record the CELL_ID values for the current region and add to the list of all CELL_IDs for all regions
        num_id_list += df_cur["CELL_ID"].tolist()

        # load the unactivated embedding values for the current region and add to the layer chunk
        df_embedding = pd.read_csv(embedding_dir+"/linear1/linear1_"+cur_region+".csv",
                                header=0)

        assert df_cur.shape[0]==df_embedding.shape[0]

        layer_chunk += df_embedding.to_numpy().tolist()

        if i%100 == 0:
            print("done with index i = "+str(i))


    assert len(num_id_list)==len(layer_chunk)

    len(layer_chunk)

    X = np.array(layer_chunk)
    X.shape

    current_time = datetime.now()
    print(current_time)
    k_means = KMeans(n_clusters=n_kmeans_clusters, random_state=0, n_init="auto").fit(X)
    current_time = datetime.now()
    print(current_time)

    print(k_means.inertia_)

    assert len(set(num_id_list)) == len(num_id_list), "duplicated cell IDs (even if not in the same image) are not allowed"

    ### Output the correspondance between CELL_ID and kmeans cluster assignment

    k_means_pred = k_means.labels_
    k_means_pred.shape

    k_means_pred = k_means_pred.tolist()
    k_means_pred[:10]

    print(len(set(k_means_pred)))
    list(set(k_means_pred))[:10]

    max(Counter(k_means_pred).values())
    min(Counter(k_means_pred).values())

    len(num_id_list)

    num_id_list[:10]

    df_kmeans = pd.DataFrame(list(zip(num_id_list, k_means_pred)), 
                            columns=["CELL_ID", "kmeans_cluster"])
    df_kmeans.shape

    df_kmeans.to_csv(output_dir+"/kmeans_cluster.csv", index=False)
    df_kmeans[:6]

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Finish running K-means on embeddings: ", current_time)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    run_k_means(**vars(args))
