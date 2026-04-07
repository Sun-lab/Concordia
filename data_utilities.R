# a function providing a list of information for each dataset
# data_name


data_features <- function(data_name){
  
  # Lung cancer dataset
  if (data_name=="cords_2024"){
    
    df_images = read.csv(paste0("./data/Cords_data/region_list.csv"),
                         header=TRUE)
    
    raw_dir = "../data/Cords_data/raw_data"
    result_subfolder = data_name
    
    n_cells_threshold = 30
    # the number of images threshold for deciding for each pair of clusters
    # whether to aggregate the distances across images
    # only aggregate if there are at least n_images_threshold images with 
    # at least n_cells_threshold in each of the two clusters
    n_images_threshold = 30
    
  }
  
  return_list = list("df" = df_images, 
                     "raw_dir"=raw_dir, 
                     "result_subfolder"=result_subfolder, 
                     "n_cells_threshold"=n_cells_threshold, 
                     "n_images_threshold"=n_images_threshold)
  
  return(return_list)
  
}