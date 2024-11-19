import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os.path as osp
import os

'''
# Requirements:
# edges.csv format:
source,target
0,1
0,2
1,2
...

# node_features.csv format:
feature1,feature2,feature3,...
0.1,0.2,0.3,...
0.2,0.3,0.4,...
...

# node_labels.csv format:
label
0
1
2
...
'''

'''
# Usage:
dataset = CustomDataset(root='path/to/data')
data = dataset[0]
'''

class CustomArxivDataset(Dataset):
    """Custom dataset for arxiv papers with different embedding options"""
    def __init__(self, root, embedding_type='scibert', transform=None, pre_transform=None):
        self.embedding_type = embedding_type
        
        # Always remove processed files before loading
        processed_dir = osp.join(root, 'processed')
        if osp.exists(processed_dir):
            for file in os.listdir(processed_dir):
                os.remove(osp.join(processed_dir, file))
        
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        # Define feature file based on embedding type
        feature_file = 'reduced_features_strict.csv' if self.embedding_type == 'scibert' else 'w2v_features_strict.csv'
        return ['edges.csv', feature_file, 'node_labels.csv']
    
    @property
    def processed_file_names(self):
        # Different processed files for different embeddings
        return [f'data_{self.embedding_type}.pt']
    
    def process(self):
        # Read edge data
        edges_df = pd.read_csv(osp.join(self.raw_dir, 'edges.csv'))
        
        # Read node features based on embedding type
        feature_file = 'reduced_features_strict.csv' if self.embedding_type == 'scibert' else 'w2v_features_strict.csv'
        features_df = pd.read_csv(osp.join(self.raw_dir, feature_file))
        
        # Read labels and splits
        labels_df = pd.read_csv(osp.join(self.raw_dir, 'node_labels.csv'))
        train_idx = np.load(osp.join(self.raw_dir, 'train_idx.npy'))
        val_idx = np.load(osp.join(self.raw_dir, 'valid_idx.npy'))
        test_idx = np.load(osp.join(self.raw_dir, 'test_idx.npy'))
        
        # Convert to PyG format
        edge_index = torch.tensor([edges_df['source'].values, edges_df['target'].values], dtype=torch.long)
        x = torch.tensor(features_df.values, dtype=torch.float)
        y = torch.tensor(labels_df['label'].values, dtype=torch.long)
        
        # Create masks
        num_nodes = x.size(0)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        # Create and save data object
        data = Data(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        torch.save(data, self.processed_paths[0])
    
    def len(self):
        return 1
        
    def get(self, idx):
        return torch.load(self.processed_paths[0])