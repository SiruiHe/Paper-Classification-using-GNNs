import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os.path as osp

'''
# Requirements:
# edges.csv 格式:
source,target
0,1
0,2
1,2
...

# node_features.csv 格式:
feature1,feature2,feature3,...
0.1,0.2,0.3,...
0.2,0.3,0.4,...
...

# node_labels.csv 格式:
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
        return ['edges.csv', 'node_features.csv', 'node_labels.csv']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        # 读取CSV文件
        edges_df = pd.read_csv(osp.join(self.raw_dir, 'edges.csv'))
        features_df = pd.read_csv(osp.join(self.raw_dir, 'node_features.csv'))
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
        
        # 创建训练/验证/测试掩码
        num_nodes = x.size(0)
        # 随机划分数据集 (可以根据需要调整比例)
        indices = torch.randperm(num_nodes)
        train_idx = indices[:int(0.8 * num_nodes)]
        val_idx = indices[int(0.8 * num_nodes):int(0.9 * num_nodes)]
        test_idx = indices[int(0.9 * num_nodes):]
        
        # 创建掩码
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