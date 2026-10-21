#!/usr/bin/env python
# coding: utf-8

# Move the graph files and composition files for processed Lermi 2025 images

# In[1]:


import os
import shutil

import pandas as pd

import sys
from data_utilities import data_features, make_model_run_id


# #### Load the list of ROIs to keep

# In[3]:


data_name = "lermi_2025"
data_dir = "../../data/Lermi_data"
graph_type = "extended"
group_scheme = "default"
dist_cutoffs = [16]
top_k = 4
ctg_comp_dist_cutoff = 0.176
degree_limit = 20


# In[4]:


graph_runs = []

for dist_cutoff in dist_cutoffs:
    expanded_edge_cutoff = 3 * dist_cutoff
    model_run_id = make_model_run_id(
        data_name, group_scheme, dist_cutoff, expanded_edge_cutoff,
        top_k, ctg_comp_dist_cutoff, degree_limit
    )
    data_dicts = data_features(
        data_name, graph_type,
        group_scheme=group_scheme,
        dist_cutoff=dist_cutoff,
        expanded_edge_cutoff=expanded_edge_cutoff,
        top_k=top_k,
        ctg_comp_dist_cutoff=ctg_comp_dist_cutoff,
        degree_limit=degree_limit,
        model_run_id=model_run_id,
        data_dir=data_dir,
    )
    train_images = sorted(data_dicts.train_images)
    graph_runs.append({
        "dist_cutoff": dist_cutoff,
        "model_run_id": model_run_id,
        "train_images": train_images,
        "source_folder": os.path.join(data_dir, "graph_objects_prepare", model_run_id),
        "target_folder": os.path.join(data_dir, "graph_objects", model_run_id),
    })


# In[5]:


{run["model_run_id"]: len(run["train_images"]) for run in graph_runs}


# #### Loop through all images/regions of interest
# 
# The file resources are under the folder:
# 
#     ../../data/Lermi_data/graph_objects_prepare/<model_run_id>
# 
# Move them to the folder
# 
#     ../../data/Lermi_data/graph_objects/<model_run_id>
# 
# This is repeated for distance cutoff 16.

# In[6]:


#from pathlib import Path

def move_graph_artifact(subfolder, extension):
    for run in graph_runs:
        target_artifact_folder = os.path.join(run["target_folder"], subfolder)
        os.makedirs(target_artifact_folder, exist_ok=True)

        for region_id in run["train_images"]:
            source_path = os.path.join(
                run["source_folder"], region_id, subfolder, f"{region_id}.{extension}"
            )
            #if Path(source_path).is_file():
            target_path = os.path.join(target_artifact_folder, f"{region_id}.{extension}")
            shutil.move(source_path, target_path)


# In[7]:


move_graph_artifact("tg_graph_basic", "gpt")


# In[8]:


move_graph_artifact("tg_graph_extended", "gpt")


# In[9]:


move_graph_artifact("tg_graph_local", "gpt")


# In[10]:


move_graph_artifact("group_composition_2nd_basic", "csv")


# In[11]:


move_graph_artifact("graph", "gpkl")


# In[12]:


{run["model_run_id"]: len(run["train_images"]) for run in graph_runs}



