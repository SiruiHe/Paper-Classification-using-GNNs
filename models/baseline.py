import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, Dropout
import pytorch_lightning as pl
from torchmetrics import Accuracy

class GCNBaseline(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # 初始化GCN层
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        # Dropout层
        self.dropout = Dropout(dropout)
        
        # 评估指标
        self.train_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        
        # 计算训练准确率
        preds = out[batch.train_mask].argmax(dim=-1)
        self.train_acc(preds, batch.y[batch.train_mask])
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        val_loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
        
        # 计算验证准确率
        preds = out[batch.val_mask].argmax(dim=-1)
        self.val_acc(preds, batch.y[batch.val_mask])
        
        self.log('val_loss', val_loss, prog_bar=True, batch_size=batch.batch_size)
        self.log('val_acc', self.val_acc, prog_bar=True, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        
        # 计算测试准确率
        preds = out[batch.test_mask].argmax(dim=-1)
        self.test_acc(preds, batch.y[batch.test_mask])
        
        self.log('test_loss', F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask]), batch_size=batch.batch_size)
        self.log('test_acc', self.test_acc, batch_size=batch.batch_size)
        return F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        return optimizer
    
    

class MLPBaseline(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # 初始化MLP层
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))
        
        # Dropout层
        self.dropout = Dropout(dropout)
        
        # 评估指标
        self.train_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)

    def forward(self, x, edge_index):
        # MLP不使用edge_index，保持接口一致只是为了兼容性
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        
        preds = out[batch.train_mask].argmax(dim=-1)
        self.train_acc(preds, batch.y[batch.train_mask])
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        val_loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
        
        preds = out[batch.val_mask].argmax(dim=-1)
        self.val_acc(preds, batch.y[batch.val_mask])
        
        self.log('val_loss', val_loss, prog_bar=True, batch_size=batch.batch_size)
        self.log('val_acc', self.val_acc, prog_bar=True, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        
        preds = out[batch.test_mask].argmax(dim=-1)
        self.test_acc(preds, batch.y[batch.test_mask])
        
        self.log('test_loss', F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask]), batch_size=batch.batch_size)
        self.log('test_acc', self.test_acc, batch_size=batch.batch_size)
        return F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        return optimizer