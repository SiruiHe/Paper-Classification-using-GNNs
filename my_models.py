import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
import dgl
from gat_dgl import GAT as DGLGATModel  # Add DGL version of GAT

import torch
import numpy as np
from typing import Tuple, Optional


class GraphPreprocessor:
    """Graph preprocessing methods comparison for ogbn-arxiv like datasets"""
    
    @staticmethod
    def basic_preprocess(edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Basic preprocessing as in current implementation
        - Convert to undirected
        - Add self-loops
        - Symmetric normalization
        """
        # Convert to undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = torch.unique(edge_index, dim=1)
        
        # Add self-loops
        num_nodes = edge_index.max().item() + 1
        diagonal_edge_index = torch.arange(num_nodes, device=edge_index.device)
        diagonal_edge_index = diagonal_edge_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, diagonal_edge_index], dim=1)
        
        # Symmetric normalization
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        row, col = edge_index
        deg = torch.bincount(row)
        deg_inv_sqrt = torch.pow(deg.float(), -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        return edge_index, edge_weight
    
    @staticmethod
    def arxiv_specific_preprocess(edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Arxiv-specific preprocessing considering its characteristics
        - Keep directed (citations have direction)
        - Add reverse edges with different weights
        - Add self-loops
        - Asymmetric normalization
        """
        num_nodes = edge_index.max().item() + 1
        
        # Add reverse edges with different weights
        reverse_edge_index = edge_index.flip(0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        
        # Create different weights for original and reverse edges
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_weight[edge_index.size(1)//2:] = 0.5  # Lower weight for reverse edges
        
        # Add self-loops
        diagonal_edge_index = torch.arange(num_nodes, device=edge_index.device)
        diagonal_edge_index = diagonal_edge_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, diagonal_edge_index], dim=1)
        edge_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=edge_index.device)], dim=0)
        
        # Asymmetric normalization (considering in-degree and out-degree differently)
        row, col = edge_index
        in_deg = torch.bincount(col, weights=edge_weight)
        out_deg = torch.bincount(row, weights=edge_weight)
        
        in_deg_inv = 1.0 / in_deg.clamp(min=1.)
        out_deg_inv = 1.0 / out_deg.clamp(min=1.)
        
        edge_weight = edge_weight * in_deg_inv[col] * out_deg_inv[row]
        
        return edge_index, edge_weight


class BaseModel(pl.LightningModule):
    """Base model containing common functionality for all models"""
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)
        
        # Best metrics tracking
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.current_loss = 0.0
        
        # Dropout layer
        self.dropout = Dropout(dropout)

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        
        # Update metrics
        preds = out[batch.train_mask].argmax(dim=-1)
        self.train_acc(preds, batch.y[batch.train_mask])
        self.current_loss = loss.item()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        
        # Validation metrics
        val_loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
        val_preds = out[batch.val_mask].argmax(dim=-1)
        self.val_acc(val_preds, batch.y[batch.val_mask])
        
        # Test metrics (calculated during validation)
        test_preds = out[batch.test_mask].argmax(dim=-1)
        self.test_acc(test_preds, batch.y[batch.test_mask])
        
        # Update best metrics
        current_val_acc = self.val_acc.compute().item()
        current_test_acc = self.test_acc.compute().item()
        
        if current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.best_test_acc = current_test_acc
        
        # Log all metrics
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('test_acc', self.test_acc, prog_bar=True)
        self.log('best_val_acc', self.best_val_acc, prog_bar=True)
        self.log('best_test_acc', self.best_test_acc, prog_bar=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask])
        
        # Calculate test accuracy
        preds = out[batch.test_mask].argmax(dim=-1)
        self.test_acc(preds, batch.y[batch.test_mask])
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

class MLPBaseline(BaseModel):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__(in_channels, out_channels, dropout)
        
        # Initialize MLP layers
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # MLP doesn't use edge_index
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=1)

class GNNBase(BaseModel):
    """Base class for GNN models with preprocessing support"""
    def __init__(self, in_channels, out_channels, dropout=0.5, preprocess_mode='basic'):
        super().__init__(in_channels, out_channels, dropout)
        self.preprocess_mode = preprocess_mode
        self.preprocessor = GraphPreprocessor()
        self.cached_edge_index = None
        self.cached_edge_weight = None

    def preprocess_graph(self, edge_index):
        if self.cached_edge_index is not None and torch.equal(edge_index, self.cached_edge_index):
            return self.cached_edge_index, self.cached_edge_weight
            
        if self.preprocess_mode == 'basic':
            edge_index, edge_weight = self.preprocessor.basic_preprocess(edge_index)
        else:
            edge_index, edge_weight = self.preprocessor.arxiv_specific_preprocess(edge_index)
            
        self.cached_edge_index = edge_index
        self.cached_edge_weight = edge_weight
        
        return edge_index, edge_weight

class GCN(GNNBase):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, preprocess_mode='basic'):
        super().__init__(in_channels, out_channels, dropout, preprocess_mode)
        
        # Initialize GCN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        edge_index, edge_weight = self.preprocess_graph(edge_index)
        
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class GAT(GNNBase):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.5, heads=4, preprocess_mode='basic'):
        super().__init__(in_channels, out_channels, dropout, preprocess_mode)
        
        # Initialize GAT layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        )

    def forward(self, x, edge_index):
        edge_index, _ = self.preprocess_graph(edge_index)
        
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
class GATAdapter(GNNBase):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.5, heads=4, attn_drop=0.05, norm='both', preprocess_mode='basic'):
        super().__init__(in_channels, out_channels, dropout, preprocess_mode)
        
        self.model = DGLGATModel(
            in_feats=in_channels,
            n_classes=out_channels,
            n_hidden=hidden_channels,
            n_layers=num_layers,
            n_heads=heads,
            activation=F.elu,
            dropout=dropout,
            attn_drop=attn_drop,
            norm=norm
        )

    def forward(self, x, edge_index):
        edge_index, _ = self.preprocess_graph(edge_index)
        g = dgl.graph((edge_index[0], edge_index[1]))
        g = g.to(x.device)
        
        out = self.model(g, x)
        return F.log_softmax(out, dim=1)


class GraphSAGE(GNNBase):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.5, preprocess_mode='basic'):
        super().__init__(in_channels, out_channels, dropout, preprocess_mode)
        
        # Initialize GraphSAGE layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # Add intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        # Add output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # Preprocess the graph structure
        edge_index, _ = self.preprocess_graph(edge_index)
        
        # Apply GraphSAGE layers with ReLU and dropout
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
        # Final layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)