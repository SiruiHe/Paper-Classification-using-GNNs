import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os.path as osp

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
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        # 列出所有原始CSV文件
        # return ['edges.csv', 'node_features_strict.csv', 'node_labels.csv']
        return ['edges.csv', 'node_features_w2v.csv', 'node_labels.csv']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        # 读取CSV文件
        edges_df = pd.read_csv(osp.join(self.raw_dir, 'edges.csv'))
        # features_df = pd.read_csv(osp.join(self.raw_dir, 'node_features_strict.csv'))
        features_df = pd.read_csv(osp.join(self.raw_dir, 'node_features_w2v.csv'))
        labels_df = pd.read_csv(osp.join(self.raw_dir, 'node_labels.csv'))
        
        # 转换边信息为COO格式
        edge_index = torch.tensor([
            edges_df['source'].values,
            edges_df['target'].values
        ], dtype=torch.long)
        
        # 转换节点特征
        x = torch.tensor(features_df.values, dtype=torch.float)
        
        # 转换标签
        y = torch.tensor(labels_df['label'].values, dtype=torch.long)
        
        # 读取预先划分的训练/验证/测试集索引
        train_idx = np.load(osp.join(self.raw_dir, 'train_idx.npy'))
        val_idx = np.load(osp.join(self.raw_dir, 'valid_idx.npy'))
        test_idx = np.load(osp.join(self.raw_dir, 'test_idx.npy'))
        
        # 创建掩码
        num_nodes = x.size(0)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        # 创建Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        torch.save(data, self.processed_paths[0])
    
    def len(self):
        return 1
        
    def get(self, idx):
        data = torch.load(self.processed_paths[0])
        return data