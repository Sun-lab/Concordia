import pandas as pd
from collections import defaultdict

class data_features(object):
    def __init__(self, data_name, graph_type):

        self.result_subfolder = data_name

        if graph_type=="extended":
            self.processed_folder_name = "tg_graph_extended"
        elif graph_type=="basic":
            self.processed_folder_name = "tg_graph_basic"
        elif graph_type=="local":
            self.processed_folder_name = "tg_graph_local"

        if data_name == "cords_2024":

            self.raw_dir = "./data/Cords_data/raw_data"
            self.dataset_root = "./data/Cords_data/graph_objects"

            # mappping each fine scale cell type to a unique integer between 0 and number of cell types - 1
            self.cell_type_mapping = {'Bcell': 0,
                                        'Blood': 1,
                                        'CD4': 2,
                                        'CD4_Treg': 3,
                                        'CD8': 4,
                                        'Collagen_CAF': 5,
                                        'HEV': 6,
                                        'IDO_CAF': 7,
                                        'IDO_CD4': 8,
                                        'IDO_CD8': 9,
                                        'Lymphatic': 10,
                                        'Myeloid': 11,
                                        'Neutrophil': 12,
                                        'Other': 13,
                                        'PD1_CD4': 14,
                                        'PDPN_CAF': 15,
                                        'SMA_CAF': 16,
                                        'TCF1/7_CD4': 17,
                                        'TCF1/7_CD8': 18,
                                        'dCAF': 19,
                                        'hypoxic': 20,
                                        'hypoxic_CAF': 21,
                                        'hypoxic_tpCAF': 22,
                                        'iCAF': 23,
                                        'ki67_CD4': 24,
                                        'ki67_CD8': 25,
                                        'mCAF': 26,
                                        'normal': 27,
                                        'tpCAF': 28,
                                        'vCAF': 29}

            # group fine scale cell types to coarse cell types
            self.group_ct_mapping = defaultdict(set)

            self.group_ct_mapping["immune"] = set(['Bcell',
                                                'CD4',
                                                'CD4_Treg',
                                                'CD8',
                                                'IDO_CD4',
                                                'IDO_CD8',
                                                'ki67_CD4',
                                                'ki67_CD8',
                                                'Myeloid',
                                                'Neutrophil',
                                                'PD1_CD4',
                                                'TCF1/7_CD4',
                                                'TCF1/7_CD8'])

            self.group_ct_mapping["tumor"] = set(['hypoxic',
                                                'normal'])

            self.group_ct_mapping["Fibroblast"] = set(['Collagen_CAF',
                                                        'dCAF',
                                                        'hypoxic_CAF',
                                                        'hypoxic_tpCAF',
                                                        'iCAF',
                                                        'IDO_CAF',
                                                        'mCAF',
                                                        'PDPN_CAF',
                                                        'SMA_CAF',
                                                        'tpCAF',
                                                        'vCAF'])

            self.group_ct_mapping["vessel"] = set(['Blood',
                                                    'HEV',
                                                    'Lymphatic'])

            self.group_ct_mapping["Other"] = set(['Other'])

            # load the list of region IDs
            df_regions = pd.read_csv("./"+ \
                                    "data/Cords_data/region_list.csv",
                                    header=0)

            self.train_images = df_regions["region_ID"].tolist()

            # distance cutoff for getting edges in basic graph
            self.dist_cutoff = 16
            # path purity cutoff for telling whether the shortest path between two cells is a qualified candidate edges to add
            self.path_purity_cutoff = 0.90
            # the cutoff for the max length of the shortest paths to consider as candidate
            # current version does not use it to filter paths
            # it should be set to a number larger than the max of number of cells in each image/tissue/region
            # where the max is taken across all images/tissues/regions in the dataset
            self.path_len_cutoff = 30000
            # a threshold for whether two clusters in an image are qualified to
            # have the embedding distance and physical distance between them computed
            # in order to be qualified, each of the two clusters must has at least this number of cells in the given image
            self.n_cells_threshold = 30

