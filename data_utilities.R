# a function providing a list of information for each dataset
# data_name


data_features <- function(data_name, model_run_id=NULL, data_dir=NULL){
  
  # Lung cancer dataset
  if (data_name=="cords_2024"){
    
    if (is.null(data_dir)){
      data_dir = "./data/Cords_data"
    }
    
    df_images = read.csv(file.path(data_dir, "region_list.csv"),
                         header=TRUE)
    
    raw_dir = file.path(data_dir, "raw_data")
    if (is.null(model_run_id)){
      result_subfolder = data_name
    }else{
      result_subfolder = model_run_id
    }
    
    n_cells_threshold = 30
    # the number of images threshold for deciding for each pair of clusters
    # whether to aggregate the distances across images
    # only aggregate if there are at least n_images_threshold images with 
    # at least n_cells_threshold in each of the two clusters
    n_images_threshold = 30
  }else{
    
    if (is.null(data_dir)){
      data_dir = file.path("./data", data_name)
    }
    
    df_images = read.csv(file.path(data_dir, "region_list.csv"),
                         header=TRUE)
    
    raw_dir = file.path(data_dir, "raw_data")
    if (is.null(model_run_id)){
      result_subfolder = data_name
    }else{
      result_subfolder = model_run_id
    }
    
    n_cells_threshold = 30
    n_images_threshold = 1
    
  }
  
  return_list = list("df" = df_images, 
                     "raw_dir"=raw_dir, 
                     "result_subfolder"=result_subfolder, 
                     "n_cells_threshold"=n_cells_threshold, 
                     "n_images_threshold"=n_images_threshold)
  
  return(return_list)
  
}
