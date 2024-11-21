import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from tuned_baseline import *
from dataset.create_dataset_best import CustomArxivDataset
from torch_geometric.loader import DataLoader
import gc
import argparse

def main():

    parser = argparse.ArgumentParser(description="Select model type and configure parameters for training.")
    parser.add_argument("--model", type=str, default="mlp", 
                        choices=["mlp", "gcn", "gat", "graphsage"], 
                        help="Type of model to use: mlp, gcn, gat, or graphsage")
    parser.add_argument("--hidden_channels", type=int, default=256, help="Number of hidden channels")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the model")
    parser.add_argument("--dropout", type=float, default=0.55, help="Dropout rate")
    parser.add_argument("--heads", type=int, default=1, help="attention heads")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()        
    gc.collect()
    pl.seed_everything(42)

    dataset = CustomArxivDataset(root='dataset/')
    #dataset.process()
    data = dataset[0]
    data = data.to(device)
    print(f"\nFeature dimensionality: {data.x.shape[1]}")
    
    params = {
        "in_channels": data.x.size(1),
        "hidden_channels": args.hidden_channels,
        "out_channels": dataset.num_classes,
        "learning_rate": args.learning_rate,
        "wd": args.wd,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    

    model_type =args.model
    print(f"Selected model: {model_type}")
    if model_type == "gcn":
        model = GCNBaseline(**params).to(device)
    elif model_type == "gat":
        model = GATBaseline(**params, heads=args.heads).to(device)   
    elif model_type == "graphsage":
        model = GraphSAGEBaseline(**params).to(device)   
    elif model_type == "mlp":
        model = MLPBaseline(**params).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    callbacks = [
        ModelCheckpoint(monitor='val_acc'),
        EarlyStopping(monitor='val_acc', patience=30, mode='max')
    ]
    
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='gpu'if torch.cuda.is_available() else 'cpu',
        devices=[0] if torch.cuda.is_available() else None,
        callbacks=callbacks,
        enable_progress_bar=True,
        precision='16-mixed', 
        gradient_clip_val=0.2
    )
    
    train_loader = DataLoader([data],batch_size=64)
    val_loader = DataLoader([data],batch_size=64)
    test_loader = DataLoader([data],batch_size=64)
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == '__main__':
    main()