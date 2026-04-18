
# merge clusters to domains, based on distance matrix


args = commandArgs(trailingOnly=TRUE)
args

if (length(args) != 4) {
  message("four arguments are expected.\n")
  quit(save="no")
}else{
  eval(parse(text=args[[1]]))
  eval(parse(text=args[[2]]))
  eval(parse(text=args[[3]]))
  eval(parse(text=args[[4]]))
}

n_kmeans_clusters = as.integer(n_kmeans_clusters)
n_domains = as.integer(n_domains)

data_name
graph_type
n_kmeans_clusters
n_domains

library(tidyr)
library(dplyr)
library(data.table)
library(ggplot2)
library(ggpubr)
library(ggpointdensity)
theme_set(theme_classic())

library(ggExtra)
library(stringr)
library(gtools)
library(RColorBrewer)

options(bitmapType = "cairo") 

source("data_utilities.R")

source("_get_n_cells_in_clusters.R")
source("_save_domain_annotations.R")


feature_list = data_features(data_name)

df = feature_list$df  
print(head(df))

# get the vector of all images considered
if (data_name=="cords_2024"){
  region_ids = df$region_ID
}

print(length(unique(region_ids)))

result_subfolder = feature_list$result_subfolder
raw_dir = feature_list$raw_dir

n_cells_threshold = feature_list$n_cells_threshold
n_images_threshold = feature_list$n_images_threshold

dir = file.path("./results", result_subfolder, graph_type)

# from the file _get_n_cells_in_clusters.R
get_n_cells_in_clusters(raw_dir, dir, region_ids, n_kmeans_clusters)

# from the file _save_domain_annotations.R
save_domain_annotations(dir, region_ids, n_kmeans_clusters, 
                        n_cells_threshold, n_images_threshold, 
                        n_domains)




gc()
sessionInfo()

q(save="no")
