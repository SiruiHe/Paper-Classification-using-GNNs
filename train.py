import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.baseline import *
from construct_dataset.create_dataset import CustomArxivDataset
from torch_geometric.loader import DataLoader, NeighborLoader

def main():
    # Load dataset
    dataset = CustomArxivDataset(root='construct_dataset/')
    data = dataset[0]
    
    # Print dataset statistics
    print(f'Dataset Statistics:')
    print(f'Number of nodes: {data.x.size(0)}')
    print(f'Number of edges: {data.edge_index.size(1)}')
    
    model_type = "gcn"  # or "gcn"
    print(f"Selected model: {model_type}")
    
    if model_type == "gcn":
        model = GCNBaseline(
            in_channels=data.x.size(1),  # 输入特征维度
            hidden_channels=256,         # 隐藏层维度
            out_channels=dataset.num_classes,  # 输出类别数
            num_layers=3,               # GCN层数
            dropout=0.5                 # dropout率
        )
    else:
        model = MLPBaseline(
            in_channels=data.x.size(1),
            hidden_channels=256,
            out_channels=dataset.num_classes,
            num_layers=3,
            dropout=0.5
        )
    
    # 设置回调函数
    callbacks = [
        ModelCheckpoint(
            monitor='val_acc',
            dirpath='checkpoints',
            filename='gcn-{epoch:02d}-{val_acc:.2f}',
            save_top_k=3,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_acc',
            patience=10,
            mode='max'
        )
    ]
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator='auto',  # 自动选择GPU或CPU
        devices=1,
        callbacks=callbacks,
        default_root_dir='logs'
    )
    
    # 创建数据加载器
    train_loader = DataLoader([data], batch_size=1)
    val_loader = DataLoader([data], batch_size=1)
    test_loader = DataLoader([data], batch_size=1)
    

    # # 使用NeighborLoader或RandomNodeSampler来进行mini-batch训练
    # train_loader = NeighborLoader(
    #     data,
    #     num_neighbors=[10, 10, 10],  # 每层采样的邻居数
    #     batch_size=128,              # 每个batch的节点数
    #     input_nodes=data.train_mask, # 训练集节点
    #     num_workers=19
    # )
    
    # val_loader = NeighborLoader(
    #     data,
    #     num_neighbors=[10, 10, 10],
    #     batch_size=128,
    #     input_nodes=data.val_mask,
    #     num_workers=19
    # )
    
    # test_loader = NeighborLoader(
    #     data,
    #     num_neighbors=[10, 10, 10],
    #     batch_size=128,
    #     input_nodes=data.test_mask,
    #     num_workers=19
    # )
    
    # 训练模型
    trainer.fit(model, train_loader, val_loader)
    
    # 测试模型
    trainer.test(model, test_loader)

if __name__ == '__main__':
    main()