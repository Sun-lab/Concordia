#!/bin/bash

# train the model on cords 2024 dataset using extended graph

data_name=cords_2024
graph_type=extended

python  train.py \
    --data_name ${data_name} \
    --graph_type ${graph_type} \
    --device gpu \
    --s_dim2 40 \
    --batch_size 64
