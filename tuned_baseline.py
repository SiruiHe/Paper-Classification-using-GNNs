import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphNorm
from torch.nn import Linear, Dropout, LayerNorm, BatchNorm1d
import pytorch_lightning as pl
from torchmetrics import Accuracy

class BaselineModel(pl.LightningModule):
    """Base class for all models"""
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate, wd, num_layers=2, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.wd = wd
        self.train_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.gradient_checkpointing = True

    def _shared_step(self, batch, stage, metric):
        out = self(batch.x, batch.edge_index)
        mask = getattr(batch, f'{stage}_mask')
        loss = F.nll_loss(out[mask], batch.y[mask])
        
        preds = out[mask].argmax(dim=-1)
        metric(preds, batch.y[mask])
        
        self.log(f'{stage}_loss', loss, prog_bar=True, batch_size=batch.batch_size)
        self.log(f'{stage}_acc', metric, prog_bar=True, batch_size=batch.batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", self.train_acc)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val", self.val_acc)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.wd
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1
            ),
            'interval': 'step'
        }
        return [optimizer], [scheduler]

class GCNBaseline(BaselineModel):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate, wd, num_layers=2, dropout=0.5):
        super().__init__(in_channels, hidden_channels, out_channels, learning_rate, wd, num_layers, dropout)
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, normalize=True))
        self.linears.append(Linear(hidden_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True))
            self.linears.append(Linear(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))
        
        self.linears.append(Linear(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, normalize=True))
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index):
        for conv, norm, linear in zip(self.convs[:-1], self.norms, self.linears):
            x_res = x
            x = conv(x, edge_index)
            x = linear(x)
            x = norm(x)
            x = F.silu(x)
            x = self.dropout(x)
            if x_res.shape == x.shape:
                x = x + x_res
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GATBaseline(BaselineModel):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate, wd,
                 num_layers=3, dropout=0.5, heads=4):
        super().__init__(in_channels, hidden_channels, out_channels, learning_rate, wd, num_layers, dropout)
        
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.attn_dropout = 0.1
        
        self.convs.append(GATConv(
            in_channels, 
            hidden_channels, 
            heads=heads, 
            dropout=self.attn_dropout
        ))
        self.linears.append(Linear(hidden_channels * heads, hidden_channels * heads))
        self.norms.append(BatchNorm1d(hidden_channels * heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels * heads,
                hidden_channels,
                heads=heads,
                dropout=self.attn_dropout
            ))
            self.linears.append(Linear(hidden_channels * heads, hidden_channels * heads))
            self.norms.append(BatchNorm1d(hidden_channels * heads))
        self.linears.append(Linear(hidden_channels * heads, hidden_channels * heads))
        self.convs.append(GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1
        ))
        
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index):
        for i, (conv, linear, norm) in enumerate(zip(self.convs[:-1], self.linears, self.norms)):
            x_res = x
            x = conv(x, edge_index)
            x = linear(x)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)
            if x_res.shape == x.shape:
                x = x + x_res
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return [optimizer], [scheduler]

class GraphSAGEBaseline(BaselineModel):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate, wd, num_layers=2, dropout=0.5):
        super().__init__(in_channels, hidden_channels, out_channels, learning_rate, wd, num_layers, dropout)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.dropout = Dropout(dropout)
        
        self.norms = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(num_layers - 1)])
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x_res = x 
            x = conv(x, edge_index) 
            x = F.relu(x)  
            x = self.dropout(x) 

            if x_res.size(1) < x.size(1): 
                diff = x.size(1) - x_res.size(1)
                x_res = torch.cat(
                    [x_res, torch.zeros(x_res.size(0), diff, device=x_res.device)],
                    dim=1
                )
            elif x_res.size(1) > x.size(1): 
                x_res = x_res[:, :x.size(1)]

            x = self.norms[i](x + x_res)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
class MLPBaseline(BaselineModel):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate, wd, num_layers=3, dropout=0.5):
        super().__init__(in_channels, hidden_channels, out_channels, learning_rate, wd, num_layers, dropout)
        self.save_hyperparameters()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels)) 
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels)) 
        self.lins.append(Linear(hidden_channels, out_channels)) 

        self.norms = torch.nn.ModuleList([BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)])

        self.dropout = Dropout(dropout)
        self.train_acc = Accuracy(task="multiclass", num_classes=out_channels)
        self.val_acc = Accuracy(task="multiclass", num_classes=out_channels)
        self.test_acc = Accuracy(task="multiclass", num_classes=out_channels)

        self.learning_rate = learning_rate
        self.wd = wd

    def forward(self, x, edge_index=None): 
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x) 
            x = self.norms[i](x)
            x = F.relu(x) 
            x = self.dropout(x) 
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=1)
