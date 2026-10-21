#!/usr/bin/env python
# coding: utf-8

# Reorganize the graph files and composition files for processed Cords 2024 images

import os
import shutil

import pandas as pd

import sys
from data_utilities import data_features, make_model_run_id


# #### Load the list of ROIs to keep

# In[3]:


data_name = "cords_2024"
data_dir = "./data/Cords_data"
graph_type = "extended"
group_scheme = "default"
dist_cutoff = 16
expanded_edge_cutoff = 48
top_k = 4
ctg_comp_dist_cutoff = 0.176
degree_limit = 20

model_run_id = make_model_run_id(data_name, group_scheme, dist_cutoff,
                                 expanded_edge_cutoff, top_k,
                                 ctg_comp_dist_cutoff, degree_limit)


# In[4]:


data_dicts  = data_features(data_name, graph_type,
                             group_scheme=group_scheme,
                             dist_cutoff=dist_cutoff,
                             expanded_edge_cutoff=expanded_edge_cutoff,
                             top_k=top_k,
                             ctg_comp_dist_cutoff=ctg_comp_dist_cutoff,
                             degree_limit=degree_limit,
                             model_run_id=model_run_id,
                             data_dir=data_dir)
train_images = data_dicts.train_images
train_images.sort()


# In[5]:


len(train_images)


# #### Loop through all images/regions of interest
# 
# The file resources are under the folder:
# 
#     ./data/Cords_data/graph_objects_prepare/<model_run_id>
# 
# Move them to the folder
# 
#     ./data/Cords_data/graph_objects/<model_run_id>  

# In[7]:


source_folder = os.path.join(data_dir, "graph_objects_prepare", model_run_id)
target_folder = os.path.join(data_dir, "graph_objects", model_run_id)


# In[8]:


target_tg_graph_basic_folder = target_folder+"/tg_graph_basic"    

os.makedirs(target_tg_graph_basic_folder, exist_ok=True)

for region_id in train_images:
    
    source_tg_graph_basic_folder = source_folder+"/"+region_id+"/tg_graph_basic"

    source_tg_graph_basic_path = source_tg_graph_basic_folder+"/"+region_id+".gpt"

    target_tg_graph_basic_path = target_tg_graph_basic_folder+"/"+region_id+".gpt"

    shutil.move(source_tg_graph_basic_path, target_tg_graph_basic_path)


# In[9]:


target_tg_graph_folder = target_folder+"/tg_graph_extended"    

os.makedirs(target_tg_graph_folder, exist_ok=True)

for region_id in train_images:
    
    source_tg_graph_folder = source_folder+"/"+region_id+"/tg_graph_extended"

    source_tg_graph_path = source_tg_graph_folder+"/"+region_id+".gpt"

    target_tg_graph_path = target_tg_graph_folder+"/"+region_id+".gpt"

    shutil.move(source_tg_graph_path, target_tg_graph_path)


# In[8]:


target_tg_graph_1st_folder = target_folder+"/tg_graph_local"    

os.makedirs(target_tg_graph_1st_folder, exist_ok=True)

for region_id in train_images:
    
    source_tg_graph_1st_folder = source_folder+"/"+region_id+"/tg_graph_local"

    source_tg_graph_1st_path = source_tg_graph_1st_folder+"/"+region_id+".gpt"

    target_tg_graph_1st_path = target_tg_graph_1st_folder+"/"+region_id+".gpt"

    shutil.move(source_tg_graph_1st_path, target_tg_graph_1st_path)


# In[11]:


target_composition_folder = target_folder+"/group_composition_2nd_basic"

os.makedirs(target_composition_folder, exist_ok=True)

for region_id in train_images:
    
    source_composition_folder = source_folder+"/"+region_id+"/group_composition_2nd_basic"
    
    source_composition_path = source_composition_folder+"/"+region_id+".csv"
    target_composition_path = target_composition_folder+"/"+region_id+".csv"

    shutil.move(source_composition_path, target_composition_path)


# In[12]:


target_graph_folder = target_folder+"/graph"    

os.makedirs(target_graph_folder, exist_ok=True)

for region_id in train_images:
    
    source_graph_folder = source_folder+"/"+region_id+"/graph"

    source_graph_path = source_graph_folder+"/"+region_id+".gpkl"

    target_graph_path = target_graph_folder+"/"+region_id+".gpkl"

    shutil.move(source_graph_path, target_graph_path)


# In[14]:


len(train_images)




