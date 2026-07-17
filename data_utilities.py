import os
import pandas as pd
from collections import defaultdict

def _format_param(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace(".", "p")


def make_model_run_id(data_name, group_scheme, dist_cutoff,
                      expanded_edge_cutoff, top_k,
                      ctg_comp_dist_cutoff, degree_limit):
    return (
        f"{data_name}"
        f"__group_{group_scheme}"
        f"__base{_format_param(dist_cutoff)}"
        f"__rad{_format_param(expanded_edge_cutoff)}"
        f"__topk{top_k}"
        f"__ctg{_format_param(ctg_comp_dist_cutoff)}"
        f"__deg{_format_param(degree_limit)}"
    )


def get_cords_cell_type_mapping():
    # mappping each fine scale cell type to a unique integer between 0 and number of cell types - 1
    return {'Bcell': 0,
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


def get_group_ct_mapping(data_name, group_scheme, cell_type_mapping):
    group_ct_mapping = defaultdict(set)

    if group_scheme == "fine":
        for cell_type in cell_type_mapping:
            group_ct_mapping[cell_type] = set([cell_type])

    elif data_name == "cords_2024" and group_scheme == "default":
        required = set(get_cords_cell_type_mapping().keys())
        if set(cell_type_mapping.keys()) != required:
            raise ValueError(
                "group_scheme='default' for data_name='cords_2024' requires Cords cell types. "
                "Use group_scheme='fine' or add a dataset-specific grouping in data_utilities.py."
            )
        group_ct_mapping["immune"] = set(['Bcell',
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

        group_ct_mapping["tumor"] = set(['hypoxic',
                                         'normal'])

        group_ct_mapping["Fibroblast"] = set(['Collagen_CAF',
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

        group_ct_mapping["vessel"] = set(['Blood',
                                          'HEV',
                                          'Lymphatic'])

        group_ct_mapping["Other"] = set(['Other'])

    else:
        raise ValueError(
            f"Unknown group_scheme '{group_scheme}' for data_name '{data_name}'. "
            "Add a dataset-specific branch in get_group_ct_mapping() or use group_scheme='fine'."
        )

    grouped_cell_types = set().union(*group_ct_mapping.values())
    missing = set(cell_type_mapping.keys()) - grouped_cell_types
    extra = grouped_cell_types - set(cell_type_mapping.keys())
    if missing:
        raise ValueError(f"Cell types missing from group_scheme {group_scheme}: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unknown cell types in group_scheme {group_scheme}: {sorted(extra)}")

    return group_ct_mapping


class data_features(object):
    def __init__(self, data_name, graph_type, group_scheme="default",
                 dist_cutoff=16, expanded_edge_cutoff=48,
                 top_k=4, ctg_comp_dist_cutoff=0.176,
                 degree_limit=20, model_run_id=None, data_dir=None):

        if model_run_id is None:
            model_run_id = make_model_run_id(data_name, group_scheme, dist_cutoff,
                                             expanded_edge_cutoff, top_k,
                                             ctg_comp_dist_cutoff, degree_limit)

        self.model_run_id = model_run_id
        self.result_subfolder = model_run_id
        self.group_scheme = group_scheme

        if graph_type=="extended":
            self.processed_folder_name = "tg_graph_extended"
        elif graph_type=="basic":
            self.processed_folder_name = "tg_graph_basic"
        elif graph_type=="local":
            self.processed_folder_name = "tg_graph_local"

        if data_name == "cords_2024":

            self.data_dir = data_dir or "./data/Cords_data"
            self.raw_dir = os.path.join(self.data_dir, "raw_data")
            self.dataset_root = os.path.join(self.data_dir, "graph_objects", model_run_id)

            self.cell_type_mapping = get_cords_cell_type_mapping()
            self.group_ct_mapping = get_group_ct_mapping(data_name, group_scheme, self.cell_type_mapping)

            # load the list of region IDs
            df_regions = pd.read_csv(os.path.join(self.data_dir, "region_list.csv"),
                                     header=0)

            self.train_images = df_regions["region_ID"].tolist()

            # distance cutoff for getting edges in basic graph
            self.dist_cutoff = dist_cutoff
            self.expanded_edge_cutoff = expanded_edge_cutoff
            self.top_k = top_k
            self.ctg_comp_dist_cutoff = ctg_comp_dist_cutoff
            self.degree_limit = degree_limit
            # path purity cutoff for telling whether the shortest path between two cells is a qualified candidate edges to add
            self.path_purity_cutoff = 0.90
            # the cutoff for the max length of the shortest paths to consider as candidate
            # current version does not use it to filter paths
            # it should be set to a number larger than the max of number of cells in each image/tissue/region
            # where the max is taken across all images/tissues/regions in the dataset
            # self.path_len_cutoff = 30000
            # a threshold for whether two clusters in an image are qualified to
            # have the embedding distance and physical distance between them computed
            # in order to be qualified, each of the two clusters must has at least this number of cells in the given image
            self.n_cells_threshold = 30

        else:
            if data_dir is None:
                data_dir = os.path.join("./data", data_name)

            self.data_dir = data_dir
            self.raw_dir = os.path.join(self.data_dir, "raw_data")
            self.dataset_root = os.path.join(self.data_dir, "graph_objects", model_run_id)

            df_regions = pd.read_csv(os.path.join(self.data_dir, "region_list.csv"),
                                     header=0)
            self.train_images = df_regions["region_ID"].tolist()

            cell_types = set()
            for region_id in self.train_images:
                df_cur = pd.read_csv(os.path.join(self.raw_dir, f"{region_id}.csv"),
                                     header=0,
                                     usecols=["CELL_TYPE"])
                cell_types.update(df_cur["CELL_TYPE"].dropna().astype(str).unique())

            self.cell_type_mapping = {cell_type: i for i, cell_type in enumerate(sorted(cell_types))}
            self.group_ct_mapping = get_group_ct_mapping(data_name, group_scheme, self.cell_type_mapping)

            self.dist_cutoff = dist_cutoff
            self.expanded_edge_cutoff = expanded_edge_cutoff
            self.top_k = top_k
            self.ctg_comp_dist_cutoff = ctg_comp_dist_cutoff
            self.degree_limit = degree_limit
            self.path_purity_cutoff = 0.90
            self.n_cells_threshold = 30
