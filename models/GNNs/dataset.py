import torch
from ogb.nodeproppred import NodePropPredDataset

import numpy as np
import pandas as pd
from collections import defaultdict
import random
import os


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, label_list=[], split_mode="random", data_dir="", train_prop=.6, valid_prop=.15):
        split_idx = {'train': [], 'valid': [], 'test': []}
        if split_mode == "random":
            label_indices = defaultdict(list)

            # 收集每种标签的索引
            for idx, label in enumerate(label_list):
                label_indices[label].append(idx)

            # 划分数据集
            for label, indices in label_indices.items():
                random.shuffle(indices)  # 打乱索引顺序
                n = len(indices)

                # 按比例划分索引
                train_end = int(n * train_prop)
                valid_end = train_end + int(n * valid_prop)

                split_idx['train'].extend(indices[:train_end])
                split_idx['valid'].extend(indices[train_end:valid_end])
                split_idx['test'].extend(indices[valid_end:])

            for label_idx in split_idx:
                split_idx[label_idx].sort()
        elif split_mode == "year":
            split_idx['train'] = np.load(f'{data_dir}train_idx.npy')
            split_idx['test'] = np.load(f'{data_dir}test_idx.npy')
            split_idx['valid'] = np.load(f'{data_dir}valid_idx.npy')

        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, split_method, embeddings):
    """ Loader for NCDataset
        Returns NCDataset
    """
    print(dataname)
    if dataname == 'ogbn-arxiv':
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname == 'arxiv-physics':
        dataset = load_arxiv_dataset(data_dir, dataname, split_method, embeddings)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx
    dataset.load_fixed_splits = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f'File {file_path} does not exist.')


def load_arxiv_dataset(data_dir, name, split_method, embeddings):
    dataset = NCDataset(name)

    edge_file_path = f'{data_dir}edges.csv'
    check_file_exists(edge_file_path)
    edge_df = pd.read_csv(edge_file_path, header=None, skiprows=1)
    edge_df = edge_df.drop_duplicates()
    edge_array = edge_df.values.T

    if embeddings == "scibert":
        file_name = "reduced_features_strict.csv"
    elif embeddings == "word2vec":
        file_name = "node_features_w2v_strict.csv"
    else:
        raise ValueError('Error: no embeddings')
    feat_file_path = f'{data_dir}{file_name}'
    check_file_exists(feat_file_path)
    feat_df = pd.read_csv(feat_file_path, header=None, skiprows=1)
    node_feat_array = feat_df.values

    label_file_path = f'{data_dir}node_labels.csv'
    check_file_exists(label_file_path)
    label_df = pd.read_csv(label_file_path, header=None, skiprows=1)
    label_array = label_df.values.flatten()

    dataset.graph = {'edge_index': torch.as_tensor(edge_array), 'edge_feat': None,
                     'node_feat': torch.as_tensor(node_feat_array).to(torch.float), 'num_nodes': len(label_array)}

    def idx_to_tensor():
        split_idx = dataset.get_idx_split(label_array, split_method, data_dir)
        tensor_split_idx = {key: torch.as_tensor(split_idx[key]) for key in split_idx}
        return tensor_split_idx
    dataset.load_fixed_splits = idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(label_array).reshape(-1, 1)
    return dataset